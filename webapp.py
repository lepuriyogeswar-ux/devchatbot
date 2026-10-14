"""Local web interface for OmniTech AI, powered by Ollama."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import chatbot


HOST = "127.0.0.1"
PORT = 8000
INDEX_FILE = Path(__file__).with_name("index.html")
MAX_REQUEST_BYTES = 256 * 1024
MAX_MESSAGES = 40
MAX_MESSAGE_LENGTH = 12_000
MAX_CONTEXT_MESSAGES = 8
MAX_CONTEXT_CHARACTERS = 16_000
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
CHAT_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_ctx": 8192,
    "num_predict": 2048,
}
WEB_DEVELOPER_PROMPT = """You are DevMate, a helpful general-purpose AI assistant.
Answer questions across subjects, including science, mathematics, history,
writing, everyday topics, programming, and web development. Give accurate,
useful answers that match the question; explain complex topics in clear steps,
show calculations when helpful, and provide complete working code when asked.
State assumptions and uncertainty honestly, and ask a brief clarifying question
only when essential information is missing. Never claim to have performed actions
or verified results that you have not.

Do not reveal private chain-of-thought. Provide the answer and a concise
explanation or summary of the reasoning instead. For medical, legal, financial,
or safety-critical topics, give general information, note relevant limitations,
and encourage consulting a qualified professional when appropriate."""


class ChatbotHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length header.") from error
        if length <= 0:
            raise ValueError("Request body is required.")
        if length > MAX_REQUEST_BYTES:
            raise OverflowError("Request body is too large.")

        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must be valid UTF-8 JSON.") from error

    def do_GET(self):
        if self.path == "/":
            try:
                body = INDEX_FILE.read_bytes()
            except OSError as error:
                self._send_json(500, {"error": "Could not read the website file."})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/status":
            try:
                with urlopen(OLLAMA_TAGS_URL, timeout=3) as response:
                    result = json.loads(response.read().decode("utf-8"))
                installed = [
                    item["name"]
                    for item in result.get("models", [])
                    if isinstance(item, dict) and isinstance(item.get("name"), str)
                ]
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "models": installed,
                        "availableModels": chatbot.AVAILABLE_MODELS,
                    },
                )
            except (URLError, TimeoutError, OSError):
                self._send_json(
                    200,
                    {
                        "ok": False,
                        "error": "Ollama is not reachable. Start Ollama and try again.",
                        "availableModels": chatbot.AVAILABLE_MODELS,
                    },
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(
                    502, {"error": "Ollama returned an invalid model-list response."}
                )
            return

        self._send_json(404, {"error": "Not found."})

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json(404, {"error": "Not found."})
            return

        try:
            payload = self._read_json()
            model, messages = self._validate_chat(payload)
        except OverflowError as error:
            self._send_json(413, {"error": str(error)})
            return
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return

        history = [{"role": "system", "content": WEB_DEVELOPER_PROMPT}]
        recent_messages = []
        context_characters = 0
        for message in reversed(messages[-MAX_CONTEXT_MESSAGES:]):
            message_length = len(message["content"])
            if (
                recent_messages
                and context_characters + message_length > MAX_CONTEXT_CHARACTERS
            ):
                break
            recent_messages.append(message)
            context_characters += message_length
        recent_messages.reverse()
        if recent_messages and recent_messages[0]["role"] == "assistant":
            recent_messages.pop(0)
        history.extend(recent_messages)

        try:
            chunks = chatbot.stream_ollama(
                model,
                history,
                options=CHAT_OPTIONS,
                keep_alive="30m",
            )
        except RuntimeError as error:
            self._send_json(502, {"error": str(error)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if not self._write_stream_event({"type": "start"}):
            chunks.close()
            return
        try:
            for chunk in chunks:
                if not self._write_stream_event({"type": "delta", "content": chunk}):
                    chunks.close()
                    return
            self._write_stream_event({"type": "done"})
        except RuntimeError as error:
            self._write_stream_event({"type": "error", "error": str(error)})

    def _write_stream_event(self, payload):
        try:
            self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
            self.wfile.flush()
        except ConnectionError:
            return False
        return True

    @staticmethod
    def _validate_chat(payload):
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object.")

        model = payload.get("model")
        if model not in chatbot.AVAILABLE_MODELS:
            raise ValueError(
                "Choose one of the supported models: "
                + ", ".join(chatbot.AVAILABLE_MODELS)
                + "."
            )

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("At least one chat message is required.")
        if len(messages) > MAX_MESSAGES:
            raise ValueError("Conversation is too long. Start a new chat.")

        validated = []
        for item in messages:
            if not isinstance(item, dict) or item.get("role") not in (
                "user",
                "assistant",
            ):
                raise ValueError("Messages must have a user or assistant role.")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Message content must not be empty.")
            if len(content) > MAX_MESSAGE_LENGTH:
                raise ValueError("A message is too long.")
            validated.append({"role": item["role"], "content": content.strip()})
        if validated[-1]["role"] != "user":
            raise ValueError("The latest message must be from the user.")
        return model, validated


def main():
    server = ThreadingHTTPServer((HOST, PORT), ChatbotHandler)
    print("OmniTech AI website is running at http://127.0.0.1:8000")
    print("Ollama must be running locally with the selected model installed.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OmniTech AI website.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
