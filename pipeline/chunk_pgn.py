"""Split a month's PGN file into chunks of complete games."""
import io
import os
import sys
import zstandard

in_path = sys.argv[1]
out_dir = sys.argv[2]
games_per_chunk = int(sys.argv[3])

os.makedirs(out_dir, exist_ok=True)
raw = zstandard.ZstdDecompressor().stream_reader(open(in_path, "rb"), read_across_frames=True)
reader = io.TextIOWrapper(raw, encoding="utf-8")

writer = None
chunk_id = 0
games_in_chunk = 0
total_games = 0

for line in reader:
    if line.startswith("[Event "):
        if writer is None or games_in_chunk == games_per_chunk:
            if writer is not None:
                writer.close()
            path = os.path.join(out_dir, f"chunk_{chunk_id:05d}.pgn.zst")
            raw_out = zstandard.ZstdCompressor().stream_writer(open(path, "wb"))
            writer = io.TextIOWrapper(raw_out, encoding="utf-8")
            chunk_id += 1
            games_in_chunk = 0
        games_in_chunk += 1
        total_games += 1
    if writer is not None:
        writer.write(line)

writer.close()
print(f"{total_games} games written to {chunk_id} chunks")
