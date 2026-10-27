"""
маленький власний сайт для перевірки краулера (лаба 6).

Це МІЙ сайт на моєму комп'ютері (localhost), тож обходити його можна без чужого
дозволу. Він має:
  • /           — головна, з якої йдуть посилання на сторінки /p/0 … /p/N-1
                  (з --chaos ще й на чотири «зламані» адреси нижче);
  • /p/<i>      — сторінки з текстом на одну з тем; кожна веде на кілька інших;
  • /robots.txt — забороняє /private/;
  • /flaky      — перші 2 запити дають 503, далі 200 (перевірка повторів);
  • /limited    — перший запит дає 429 з Retry-After: 1, далі 200;
  • /loop-a     — редірект на /loop-b, а той назад (зациклені редіректи);
  • /slow       — відповідає через 30 секунд (перевірка таймауту);
  • --latency   — штучна затримка кожної відповіді (імітує мережу).

Запуск:
    python -m findex.demosite --port 8000 --pages 500 --latency 0.05
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOPICS = {
    "тварини": "кіт пес вовк лисиця ведмідь птах риба кінь корова вівця",
    "місто": "вулиця площа міст ринок вокзал бібліотека театр парк річка собор",
    "наука": "атом планета зірка алгоритм індекс формула дослід гіпотеза клітина енергія",
    "їжа": "хліб борщ сир молоко яблуко мед пиріг вареники чай кава",
    "спорт": "м'яч команда матч гравець тренер чемпіонат біг плавання шахи турнір",
}
TOPIC_NAMES = list(TOPICS)


class _Server(ThreadingHTTPServer):
    # черга очікування з'єднань: типові 5 замало, коли краулер відкриває 20 з'єднань одразу
    request_queue_size = 128

    def handle_error(self, request: object, client_address: object) -> None:
        # краулер при зупинці сам рве з'єднання: це не помилка сайту, тому не засмічую консоль
        pass


class DemoState:
    """Лічильники, спільні для всіх запитів (для /flaky і /limited)."""

    def __init__(self, pages: int, latency: float, chaos: bool = False) -> None:
        self.pages = pages
        self.chaos = chaos
        self.latency = latency
        self.lock = threading.Lock()
        self.flaky_calls = 0
        self.limited_calls = 0
        self.requests = 0


def ok_page(name: str) -> str:
    return f"<html><head><title>{name}</title></head><body>{name}</body></html>"


def make_page(i: int, n: int) -> str:
    """Сторінка номер i: заголовок, абзац і посилання на 3 «дочірні» сторінки."""
    rng = random.Random(i)  # одна й та сама сторінка щоразу однакова
    topic = TOPIC_NAMES[i % len(TOPIC_NAMES)]
    words = TOPICS[topic].split()
    text = " ".join(rng.choice(words) for _ in range(120))
    links = [c for c in (3 * i + 1, 3 * i + 2, 3 * i + 3) if c < n]
    links.append((i * 7 + 5) % n)  # одне посилання «вбік», щоб був цикл
    anchors = "".join(f'<a href="/p/{c}">сторінка {c}</a> ' for c in links)
    return (
        f"<html><head><title>Сторінка {i}: {topic}</title>"
        f"<style>p {{color: red}}</style><script>var x = 'не текст';</script></head>"
        f"<body><h1>{topic} {i}</h1><p>{text}</p><p>{anchors}</p>"
        f'<a href="/private/secret">секрет</a> <a href="https://example.org/">назовні</a>'
        f"</body></html>"
    )


def make_handler(state: DemoState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # тиша в консолі
            pass

        def _send(self, code: int, body: str, ctype: str = "text/html; charset=utf-8",
                  headers: dict[str, str] | None = None) -> None:  # fmt: skip
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            with state.lock:
                state.requests += 1
            if state.latency:
                time.sleep(state.latency)
            path = self.path.split("?")[0]
            if path == "/robots.txt":
                self._send(200, "User-agent: *\nDisallow: /private/\n", "text/plain")
            elif path == "/":
                items = "".join(
                    f'<li><a href="/p/{i}">{i}</a></li>' for i in range(min(5, state.pages))
                )
                special = (
                    '<a href="/flaky">flaky</a> <a href="/limited">limited</a> '
                    '<a href="/loop-a">loop</a> <a href="/slow">slow</a>'
                    if state.chaos
                    else ""
                )
                self._send(
                    200,
                    f"<html><head><title>Демо-сайт</title></head><body><ul>{items}</ul>{special}</body></html>",
                )
            elif path.startswith("/p/") and path[3:].isdigit() and int(path[3:]) < state.pages:
                self._send(200, make_page(int(path[3:]), state.pages))
            elif path == "/flaky":
                with state.lock:
                    state.flaky_calls += 1
                    n = state.flaky_calls
                if n <= 2:
                    self._send(503, "<html><title>503</title></html>")
                else:
                    self._send(
                        200,
                        ok_page("flaky ok"),
                    )
            elif path == "/limited":
                with state.lock:
                    state.limited_calls += 1
                    n = state.limited_calls
                if n == 1:
                    self._send(429, "slow down", "text/plain", {"Retry-After": "1"})
                else:
                    self._send(
                        200,
                        ok_page("limited ok"),
                    )
            elif path == "/loop-a":
                self._send(302, "", headers={"Location": "/loop-b"})
            elif path == "/loop-b":
                self._send(302, "", headers={"Location": "/loop-a"})
            elif path == "/slow":
                time.sleep(30)
                self._send(200, "<html><title>slow</title></html>")
            else:
                self._send(404, "<html><title>404</title></html>")

    return Handler


def start_server(
    port: int = 0, pages: int = 500, latency: float = 0.0, chaos: bool = False
) -> tuple[ThreadingHTTPServer, DemoState]:
    """Запускає сервер у фоновому потоці (port=0 означає «будь-який вільний»)."""
    state = DemoState(pages, latency, chaos)
    server = _Server(("127.0.0.1", port), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pages", type=int, default=500)
    ap.add_argument("--latency", type=float, default=0.05, help="затримка кожної відповіді, с")
    ap.add_argument("--chaos", action="store_true", help="додати /flaky, /limited, /loop-a, /slow")
    args = ap.parse_args()
    server, _state = start_server(args.port, args.pages, args.latency, args.chaos)
    print(f"демо-сайт: http://127.0.0.1:{server.server_port}/  (Ctrl+C щоб зупинити)")
    try:
        # wait() без таймауту на Windows не реагує на Ctrl+C, тому чекаю короткими шматками
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
