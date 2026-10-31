#!/usr/bin/env python3
"""
AirCompute Terminal Chat & Swarm Mesh Monitor
Sends a prompt to the decentralized swarm, streams response tokens live in the terminal,
and monitors real-time Mesh Inspector telemetry logs.
"""
import asyncio
import json
import sys
import websockets
import httpx

async def run_terminal_test(prompt: str):
    print("=" * 70)
    print("🌐 AIRCOMPUTE DECENTRALIZED SWARM TERMINAL CLIENT")
    print(f"📝 Prompt: \"{prompt}\"")
    print("=" * 70 + "\n")

    ws_url = "ws://127.0.0.1:8000/ws"
    trigger_url = "http://127.0.0.1:8000/api/trigger"

    async with websockets.connect(ws_url) as ws:
        async def _trigger():
            async with httpx.AsyncClient(timeout=120.0) as client:
                await client.post(trigger_url, json={"prompt": prompt})

        # Launch trigger in background task so WebSocket receives tokens concurrently
        trigger_task = asyncio.create_task(_trigger())
        print("🚀 Triggered Swarm Prompt!\n")

        print("🤖 [AI RESPONSE STREAMING]: ")
        print("-" * 50)

        tokens_received = []
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                data = json.loads(msg)

                if data.get("type") == "token":
                    token = data.get("token", "")
                    tokens_received.append(token)
                    sys.stdout.write(f"[{token}] ")
                    sys.stdout.flush()

                elif data.get("type") == "telemetry":
                    m1_cpu = data.get("m1_cpu", 0)
                    m2_cpu = data.get("m2_cpu", 0)
                    step = data.get("pipeline_step", "")
                    sys.stdout.write(f"\n   📊 [MESH TELEMETRY] Laptop CPU: {m1_cpu}% | Phone NPU: {m2_cpu}% | Active: {step}")
                    sys.stdout.flush()

                elif data.get("type") == "log":
                    log_msg = data.get("msg", "")
                    print(f"\n\n📜 {log_msg}")

        except asyncio.TimeoutError:
            print("\n⏰ Stream finished.")

        full_response = "".join(tokens_received)
        print("-" * 50)
        print(f"\n✅ Total Tokens Generated: {len(tokens_received)}")
        print(f"💬 Complete Answer: \"{full_response.strip()}\"")
        print("=" * 70)

if __name__ == "__main__":
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "What is 2 + 2?"
    asyncio.run(run_terminal_test(prompt_text))
