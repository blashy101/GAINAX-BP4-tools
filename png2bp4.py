import struct
from pathlib import Path
from PIL import Image


BP4_MAGIC = 0x88888899


def load_image_bgra(path: str) -> tuple[int, int, bytes]:
    img = Image.open(path).convert("RGBA")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    width, height = img.size
    rgba = img.tobytes()

    bgra = bytearray(len(rgba))
    for i in range(0, len(rgba), 4):
        r, g, b, a = rgba[i:i + 4]
        bgra[i:i + 4] = bytes([b, g, r, a])

    return width, height, bytes(bgra)


def get_pixel_bgra(raw: bytes, width: int, x: int, y: int) -> tuple[int, int, int, int]:
    off = (y * width + x) * 4
    b, g, r, a = raw[off:off + 4]
    return b, g, r, a


def parse_bp4_header(data: bytes) -> tuple[int, int, bytes]:
    if len(data) < 0x46:
        raise ValueError("BP4 file is too small")

    magic, width, height = struct.unpack_from("<III", data, 0)
    if magic != BP4_MAGIC:
        raise ValueError("Not a BP4 file")

    return width, height, data[:0x46]


def read_original_mode_table(data: bytes, width: int, height: int) -> bytes:
    num_blocks = (width * height) // 64
    pos = 0x46
    modes = data[pos:pos + num_blocks]
    if len(modes) != num_blocks:
        raise ValueError("Original BP4 is truncated in mode table")
    return modes


def encode_rgb_mode(block_pixels: list[tuple[int, int, int, int]], rgb_mode: int):
    bs = [p[0] for p in block_pixels]
    gs = [p[1] for p in block_pixels]
    rs = [p[2] for p in block_pixels]

    if rgb_mode == 0:
        b0, g0, r0 = bs[0], gs[0], rs[0]
        if all((b, g, r) == (b0, g0, r0) for b, g, r, _ in block_pixels):
            return b0, g0, r0, b""
        return None

    elif rgb_mode == 1:
        base_b = min(bs)
        base_g = min(gs)
        base_r = min(rs)
        out = bytearray()
        for b, g, r, _ in block_pixels:
            db = b - base_b
            dg = g - base_g
            dr = r - base_r
            if not (0 <= db <= 7 and 0 <= dg <= 7 and 0 <= dr <= 3):
                return None
            out.append(db | (dg << 3) | (dr << 6))
        return base_b, base_g, base_r, bytes(out)

    elif rgb_mode == 2:
        base_b = min(bs)
        base_g = min(gs)
        base_r = min(rs)
        out = bytearray()
        for b, g, r, _ in block_pixels:
            db = b - base_b
            dg = g - base_g
            dr = r - base_r
            if not (0 <= db <= 3 and 0 <= dg <= 7 and 0 <= dr <= 7):
                return None
            out.append(db | (dg << 2) | (dr << 5))
        return base_b, base_g, base_r, bytes(out)

    elif rgb_mode == 3:
        base_b = min(bs)
        base_g = min(gs)
        base_r = min(rs)
        out = bytearray()
        for b, g, r, _ in block_pixels:
            db = b - base_b
            dg = g - base_g
            dr = r - base_r
            if not (0 <= db <= 7 and 0 <= dg <= 3 and 0 <= dr <= 7):
                return None
            out.append(db | (dg << 3) | (dr << 5))
        return base_b, base_g, base_r, bytes(out)

    elif rgb_mode == 4:
        base_b = min(bs)
        base_g = min(gs)
        base_r = min(rs)

        vals = []
        for b, g, r, _ in block_pixels:
            db = b - base_b
            dg = g - base_g
            dr = r - base_r
            if db != dg or dg != dr or not (0 <= db <= 15):
                return None
            vals.append(db)

        out = bytearray()
        for i in range(0, len(vals), 2):
            lo = vals[i]
            hi = vals[i + 1]
            out.append(lo | (hi << 4))
        return base_b, base_g, base_r, bytes(out)

    elif rgb_mode == 5:
        out = bytearray()
        for b, g, r, _ in block_pixels:
            if not (b == g == r):
                return None
            out.append(b)
        return 0, 0, 0, bytes(out)

    elif rgb_mode == 6:
        base_b = min(bs)
        base_g = min(gs)
        base_r = min(rs)
        out = bytearray()
        for b, g, r, _ in block_pixels:
            db = b - base_b
            dg = g - base_g
            dr = r - base_r
            if not (0 <= db <= 31 and 0 <= dg <= 31 and 0 <= dr <= 31):
                return None
            lo = (db & 0x1F) | ((dg & 0x07) << 5)
            hi = ((dg >> 3) & 0x03) | ((dr & 0x1F) << 2)
            out.extend([lo, hi])
        return base_b, base_g, base_r, bytes(out)

    elif rgb_mode == 7:
        out = bytearray()
        for b, g, r, _ in block_pixels:
            out.extend([b, g, r])
        return 0, 0, 0, bytes(out)

    return None


def encode_block(block_pixels: list[tuple[int, int, int, int]], preferred_mode: int):
    preferred_alpha = preferred_mode >> 4
    preferred_rgb = preferred_mode & 0x0F

    alphas = [p[3] for p in block_pixels]
    all_alpha_same = all(a == alphas[0] for a in alphas)

    if preferred_alpha == 1 and all_alpha_same:
        alpha_mode = 1
        alpha_payload = b""
        base_a = alphas[0]
    elif preferred_alpha == 2:
        alpha_mode = 2
        alpha_payload = bytes(alphas)
        base_a = alphas[0]
    else:
        if all_alpha_same:
            alpha_mode = 1
            alpha_payload = b""
            base_a = alphas[0]
        else:
            alpha_mode = 2
            alpha_payload = bytes(alphas)
            base_a = alphas[0]

    rgb_try = encode_rgb_mode(block_pixels, preferred_rgb)

    if rgb_try is not None:
        base_b, base_g, base_r, rgb_payload = rgb_try
        mode_byte = (alpha_mode << 4) | preferred_rgb
        return mode_byte, bytes([base_b, base_g, base_r, base_a]), rgb_payload + alpha_payload

    base_b, base_g, base_r, rgb_payload = encode_rgb_mode(block_pixels, 7)
    mode_byte = (alpha_mode << 4) | 7
    return mode_byte, bytes([base_b, base_g, base_r, base_a]), rgb_payload + alpha_payload


def encode_bp4_from_template(original_bp4: str, edited_image: str, output_bp4: str) -> None:
    original_data = Path(original_bp4).read_bytes()
    orig_width, orig_height, header_template = parse_bp4_header(original_data)
    original_modes = read_original_mode_table(original_data, orig_width, orig_height)

    width, height, pixels = load_image_bgra(edited_image)

    if width != orig_width or height != orig_height:
        raise ValueError(
            f"Edited image dimensions {width}x{height} do not match original {orig_width}x{orig_height}"
        )

    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("This encoder expects width and height to be multiples of 8")

    blocks_x = width // 8
    blocks_y = height // 8

    mode_table = bytearray()
    base_table = bytearray()
    payload = bytearray()

    block_index = 0
    preserved_count = 0
    fallback_count = 0

    for by in range(blocks_y):
        for bx in range(blocks_x):
            block_pixels = []
            for y in range(8):
                for x in range(8):
                    block_pixels.append(
                        get_pixel_bgra(pixels, width, bx * 8 + x, by * 8 + y)
                    )

            preferred_mode = original_modes[block_index]
            mode_byte, base_bytes, payload_bytes = encode_block(block_pixels, preferred_mode)

            if mode_byte == preferred_mode:
                preserved_count += 1
            else:
                fallback_count += 1

            mode_table.append(mode_byte)
            base_table.extend(base_bytes)
            payload.extend(payload_bytes)

            block_index += 1

    out = bytearray()
    out += header_template
    out += mode_table
    out += base_table
    out += payload

    Path(output_bp4).write_bytes(out)

    print(f"Done. Preserved original mode on {preserved_count} blocks, fell back on {fallback_count} blocks.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage:")
        print("  python png2bp4.py original.bp4 edited.png output.bp4")
        raise SystemExit(1)

    encode_bp4_from_template(sys.argv[1], sys.argv[2], sys.argv[3])