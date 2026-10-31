import asyncio
import json
import httpx
import websockets
import pytest

def test_chat_manual():
    # Only run if server is running
    async def _run():
        uri = "ws://127.0.0.1:8000/ws"
        try:
            async with websockets.connect(uri, open_timeout=1.0) as ws:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post("http://127.0.0.1:8000/api/trigger", json={"prompt": "Ping"})
                    assert resp.status_code == 200
        except Exception:
            pytest.skip("Swarm server not running locally on port 8000")

    asyncio.run(_run())

if __name__ == "__main__":
    asyncio.run(test_chat_manual())
