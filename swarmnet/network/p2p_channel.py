"""
AirCompute P2P QUIC/UDP Transport Channel with CRC32 Packet Integrity & Selective ARQ
"""
import zlib
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class PacketHeader:
    MAGIC = 0x53574152  # 'SWAR' in ASCII
    seq_num: int
    total_chunks: int
    chunk_index: int
    payload_len: int
    crc32: int

class P2PChannel:
    HEADER_STRUCT = "!IIII I"  # Magic, SeqNum, TotalChunks, ChunkIdx, PayloadLen, CRC32
    HEADER_SIZE = struct.calcsize("!IIIII")

    def __init__(self, channel_id: str, max_chunk_size: int = 1400):
        self.channel_id = channel_id
        self.max_chunk_size = max_chunk_size
        self.next_seq_num = 1
        self.reception_buffers: Dict[int, Dict[int, bytes]] = {}
        self.rtt_history = []
        self.current_rtt_ms = 5.0

    def frame_packet(self, payload: bytes, seq_num: Optional[int] = None) -> list[bytes]:
        """
        Frames an activation payload into chunked UDP datagrams with CRC32 checksums.
        """
        if seq_num is None:
            seq_num = self.next_seq_num
            self.next_seq_num += 1

        chunks = []
        total_chunks = max(1, (len(payload) + self.max_chunk_size - 1) // self.max_chunk_size)

        for i in range(total_chunks):
            start = i * self.max_chunk_size
            end = min(len(payload), start + self.max_chunk_size)
            chunk_data = payload[start:end]
            crc = zlib.crc32(chunk_data) & 0xFFFFFFFF
            header = struct.pack(
                "!IIIII",
                PacketHeader.MAGIC,
                seq_num,
                total_chunks,
                i,
                crc
            )
            chunks.append(header + chunk_data)

        return chunks

    def process_incoming_chunk(self, raw_chunk: bytes) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """
        Processes an incoming framed chunk, validates CRC32, and reconstructs the full payload when complete.
        Returns: (is_complete, full_payload_or_none, error_code_or_none)
        """
        if len(raw_chunk) < self.HEADER_SIZE:
            return False, None, "ERR_PACKET_TOO_SHORT"

        magic, seq_num, total_chunks, chunk_idx, expected_crc = struct.unpack("!IIIII", raw_chunk[:self.HEADER_SIZE])
        if magic != PacketHeader.MAGIC:
            return False, None, "ERR_INVALID_MAGIC_HEADER"

        payload_chunk = raw_chunk[self.HEADER_SIZE:]
        actual_crc = zlib.crc32(payload_chunk) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            return False, None, "ERR_CRC32_CORRUPTION"

        if seq_num not in self.reception_buffers:
            self.reception_buffers[seq_num] = {}

        self.reception_buffers[seq_num][chunk_idx] = payload_chunk

        # Check if all chunks received
        if len(self.reception_buffers[seq_num]) == total_chunks:
            full_data = bytearray()
            for idx in range(total_chunks):
                full_data.extend(self.reception_buffers[seq_num][idx])
            del self.reception_buffers[seq_num]
            return True, bytes(full_data), None

        return False, None, None

    def update_rtt(self, sample_ms: float):
        """Exponential Moving Average (EMA) RTT calculation (alpha=0.125)"""
        self.rtt_history.append(sample_ms)
        self.current_rtt_ms = 0.875 * self.current_rtt_ms + 0.125 * sample_ms
