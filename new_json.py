#!/usr/bin/env python3
import json, argparse, sys, pathlib

def main():
    p = argparse.ArgumentParser(description="Pretty-print JSON to a new file.")
    p.add_argument("input", help="Đường dẫn file JSON nguồn")
    p.add_argument("-o", "--output", help="File đích (mặc định: <input>.pretty.json)")
    p.add_argument("--indent", type=int, default=2, help="Số khoảng trắng mỗi mức thụt (mặc định: 2)")
    p.add_argument("--sort-keys", action="store_true", help="Sắp xếp key theo alphabet")
    args = p.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.is_file():
        print(f"Không tìm thấy file: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = pathlib.Path(args.output) if args.output else in_path.with_suffix(in_path.suffix + ".pretty.json")

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON không hợp lệ: {e}", file=sys.stderr)
        sys.exit(2)

    # ensure_ascii=False để giữ ký tự Unicode; separators để thêm dấu cách sau dấu phẩy/2 chấm cho dễ đọc
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            data, f,
            indent=args.indent,
            ensure_ascii=False,
            sort_keys=args.sort_keys,
            separators=(", ", ": ")
        )
        f.write("\n")  # dòng mới cuối file (POSIX-friendly)

    print(f"Đã ghi JSON format đẹp vào: {out_path}")

if __name__ == "__main__":
    main()