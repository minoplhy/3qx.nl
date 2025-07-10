import os
import numpy as np
from PIL import Image, ImageSequence, ImageDraw, ImageFont

ASCII_CHARS = '@%#*+=-:. '  # From darkest to lightest

def get_font(font_path=None, font_size=12):
    return ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, font_size)

def calc_ascii_image_size(txt_paths, font):
    max_line_len = 0
    max_lines = 0
    all_lines = []
    for txt_path in txt_paths:
        with open(txt_path, encoding="utf8") as f:
            lines = f.read().splitlines()
        if len(lines) > max_lines:
            max_lines = len(lines)
        line_lens = [len(line) for line in lines]
        if line_lens:
            longest = max(line_lens)
            if longest > max_line_len:
                max_line_len = longest
        all_lines.append(lines)
    # get font metrics
    dummy_img = Image.new("L", (1,1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0,0), "X", font=font)
    char_width = bbox[2]-bbox[0]
    char_height = bbox[3]-bbox[1]
    img_w = char_width * max_line_len
    img_h = char_height * max_lines
    return max_line_len, max_lines, img_w, img_h, char_width, char_height


def ascii_to_image_pad(lines, max_line_len, max_lines, img_w, img_h, font, char_width, char_height, bg="white", fg="black"):
    # Pad each line to max_line_len and pad lines to max_lines
    padded_lines = [line.ljust(max_line_len) for line in lines]
    while len(padded_lines) < max_lines:
        padded_lines.append(" " * max_line_len)
    image = Image.new('RGB', (img_w, img_h), bg)
    draw = ImageDraw.Draw(image)
    for y, line in enumerate(padded_lines):
        draw.text((0, y*char_height), line, fill=fg, font=font)
    return image

def image_to_ascii(image, width=80):
    # Resize maintaining aspect ratio compensate for char height
    aspect_ratio = image.height/image.width
    new_height = int(aspect_ratio * width * 0.55)
    image = image.resize((width, new_height)).convert("L")  # Grayscale

    pixels = np.array(image)
    chars = np.array(list(ASCII_CHARS))
    norm = ((pixels - pixels.min()) / (np.ptp(pixels)+1e-9))
    ind = (norm * (len(ASCII_CHARS)-1)).astype(int)
    ascii_str = "\n".join(("".join(chars[row]) for row in ind))
    return ascii_str

def ascii_to_image(ascii_txt, font_path=None, font_size=12, bg="white", fg="black"):
    lines = ascii_txt.splitlines()
    font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, font_size)
    # Estimate image size
    dummy_img = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), "X", font=font)
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    img_w = char_width * max(map(len, lines))
    img_h = char_height * len(lines)
    image = Image.new('RGB', (img_w, img_h), bg)
    draw = ImageDraw.Draw(image)
    for y, line in enumerate(lines):
        draw.text((0, y * char_height), line, fill=fg, font=font)
    return image


def gif_to_ascii_txts(gif_path, out_dir='ascii_frames', width=80):
    os.makedirs(out_dir, exist_ok=True)
    image = Image.open(gif_path)
    duration = image.info.get('duration', 100)
    txt_paths = []
    for i, frame in enumerate(ImageSequence.Iterator(image)):
        ascii_art = image_to_ascii(frame, width)
        txt_file = os.path.join(out_dir, f"frame_{i:03d}.txt")
        with open(txt_file, "w", encoding="utf8") as f:
            f.write(ascii_art)
        txt_paths.append(txt_file)
    return txt_paths, duration

def ascii_txts_to_gif(txt_paths, out_gif_path, duration=100, font_path=None, font_size=12, fg="black", bg="white"):
    font = get_font(font_path, font_size)
    max_line_len, max_lines, img_w, img_h, char_width, char_height = calc_ascii_image_size(txt_paths, font)
    imgs = []
    for txt_path in txt_paths:
        with open(txt_path, encoding="utf8") as f:
            lines = f.read().splitlines()
        img = ascii_to_image_pad(lines, max_line_len, max_lines, img_w, img_h, font, char_width, char_height, bg=bg, fg=fg)
        imgs.append(img)
    imgs[0].save(out_gif_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

if __name__ == "__main__":
    # Set paths
    input_gif = "teto-tetoris.gif"
    ascii_dir = "ascii_frames"
    output_gif = "output_ascii.gif"
    ascii_width = 80  # You can adjust

    # Step 1: Extract ASCII .txts
    txt_files, frame_duration = gif_to_ascii_txts(input_gif, ascii_dir, ascii_width)
    print(f"ASCII frames saved to {ascii_dir}/")

    # Step 2: Convert back to GIF
    ascii_txts_to_gif(txt_files, output_gif, duration=frame_duration)
    print(f"ASCII gif saved as: {output_gif}")
