import struct
from pathlib import Path
from PIL import Image


BP4_MAGIC = 0x88888899


def decode_bp4(data: bytes) -> tuple[bytes, int, int]:
    magic, width, height = struct.unpack_from("<III", data, 0)
    if magic != BP4_MAGIC:
        raise ValueError("Not a BP4 file")

    bmp_header = bytearray(data[0x10:0x46])

    pixel_offset = struct.unpack_from("<I", bmp_header, 10)[0]
    if pixel_offset < len(bmp_header):
        raise ValueError(
            f"Invalid embedded BMP pixel offset: {pixel_offset} "
            f"(header length is {len(bmp_header)})"
        )

    image_size = width * height * 4

    struct.pack_into("<I", bmp_header, 2, pixel_offset + image_size)
    struct.pack_into("<H", bmp_header, 0x1C, 32)
    struct.pack_into("<I", bmp_header, 0x22, image_size)

    num_blocks = (width * height) // 64

    pos = 0x46
    modes = data[pos:pos + num_blocks]
    pos += num_blocks

    base_colors = data[pos:pos + num_blocks * 4]
    pos += num_blocks * 4

    out = bytearray(width * height * 4)

    bits_per_rgb_mode = {
        0: 0,
        1: 8,
        2: 8,
        3: 8,
        4: 4,
        5: 8,
        6: 16,
        7: 24,
    }

    blocks_x = (width + 7) // 8
    blocks_y = (height + 7) // 8

    block_index = 0

    for by in range(blocks_y):
        bh = min(8, height - by * 8)

        for bx in range(blocks_x):
            bw = min(8, width - bx * 8)

            mode_byte = modes[block_index]
            rgb_mode = mode_byte & 0x0F
            alpha_mode = mode_byte >> 4

            if rgb_mode not in bits_per_rgb_mode:
                raise ValueError(f"Unknown BP4 RGB mode {rgb_mode} at block {block_index}")

            base = base_colors[block_index * 4:block_index * 4 + 4]
            base_b, base_g, base_r, base_a = base

            rgb_bits = bits_per_rgb_mode[rgb_mode]
            rgb_payload_size = (bw * bh * rgb_bits + 7) // 8
            rgb_payload = data[pos:pos + rgb_payload_size]
            pos += rgb_payload_size

            alpha_values = [base_a] * (bw * bh)

            if alpha_mode == 2:
                alpha_payload_size = bw * bh
                alpha_payload = data[pos:pos + alpha_payload_size]
                pos += alpha_payload_size
                alpha_values = list(alpha_payload)
            elif alpha_mode != 1:
                raise ValueError(f"Unknown BP4 alpha mode {alpha_mode} at block {block_index}")

            values = [None] * (bw * bh)

            if rgb_bits == 4:
                for i in range(bw * bh):
                    b = rgb_payload[i // 2]
                    values[i] = (b & 0x0F) if (i % 2 == 0) else ((b >> 4) & 0x0F)
            elif rgb_bits == 8:
                values = list(rgb_payload[:bw * bh])
            elif rgb_bits == 16:
                values = [rgb_payload[i * 2:i * 2 + 2] for i in range(bw * bh)]
            elif rgb_bits == 24:
                values = [rgb_payload[i * 3:i * 3 + 3] for i in range(bw * bh)]

            for y in range(bh):
                for x in range(bw):
                    i = y * bw + x
                    dst = ((by * 8 + y) * width + (bx * 8 + x)) * 4

                    if rgb_mode == 0:
                        b, g, r = base_b, base_g, base_r
                    elif rgb_mode == 1:
                        v = values[i]
                        b = (base_b + (v & 0x07)) & 0xFF
                        g = (base_g + ((v >> 3) & 0x07)) & 0xFF
                        r = (base_r + ((v >> 6) & 0x03)) & 0xFF
                    elif rgb_mode == 2:
                        v = values[i]
                        b = (base_b + (v & 0x03)) & 0xFF
                        g = (base_g + ((v >> 2) & 0x07)) & 0xFF
                        r = (base_r + ((v >> 5) & 0x07)) & 0xFF
                    elif rgb_mode == 3:
                        v = values[i]
                        b = (base_b + (v & 0x07)) & 0xFF
                        g = (base_g + ((v >> 3) & 0x03)) & 0xFF
                        r = (base_r + ((v >> 5) & 0x07)) & 0xFF
                    elif rgb_mode == 4:
                        v = values[i]
                        b = (base_b + v) & 0xFF
                        g = (base_g + v) & 0xFF
                        r = (base_r + v) & 0xFF
                    elif rgb_mode == 5:
                        v = values[i]
                        b = g = r = v
                    elif rgb_mode == 6:
                        lo = values[i][0]
                        hi = values[i][1]
                        b = (base_b + (lo & 0x1F)) & 0xFF
                        g = (base_g + (((lo >> 5) & 0x07) + ((hi & 0x03) << 3))) & 0xFF
                        r = (base_r + ((hi & 0x7C) >> 2)) & 0xFF
                    elif rgb_mode == 7:
                        px = values[i]
                        b, g, r = px[0], px[1], px[2]

                    a = alpha_values[i]
                    out[dst:dst + 4] = bytes([b, g, r, a])

            block_index += 1

    if pos != len(data):
        raise ValueError(
            f"Did not consume entire file: stopped at 0x{pos:X}, file size 0x{len(data):X}"
        )

    gap = pixel_offset - len(bmp_header)
    bmp_data = bytes(bmp_header) + (b"\x00" * gap) + bytes(out)

    return bmp_data, width, height


def bp4_to_png(src_path: str, dst_path: str) -> None:
    data = Path(src_path).read_bytes()
    bmp, width, height = decode_bp4(data)

    pixel_offset = struct.unpack_from("<I", bmp, 10)[0]
    raw = bmp[pixel_offset:]

    expected_size = width * height * 4
    if len(raw) < expected_size:
        raise ValueError(
            f"Not enough pixel data: got {len(raw)} bytes, expected {expected_size}"
        )

    raw = raw[:expected_size]
    row_size = width * 4

    rgba = bytearray()
    for y in range(height):
        src_y = height - 1 - y
        row = raw[src_y * row_size:(src_y + 1) * row_size]

        for i in range(0, len(row), 4):
            b, g, r, a = row[i:i + 4]
            rgba.extend([r, g, b, a])

    img = Image.frombytes("RGBA", (width, height), bytes(rgba))
    img.save(dst_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage:")
        print("  python bp42png.py input.bp4 output.png")
        raise SystemExit(1)

    bp4_to_png(sys.argv[1], sys.argv[2])