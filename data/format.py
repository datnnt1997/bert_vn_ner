import csv
if __name__ == "__main__":
    with open("./train.csv", "r", encoding="utf-8") as f:
        pos_tags = set()
        chunk_tags = set()
        lines = f.readlines()
        for line in lines:
            items = line.split("\t")
            if len(items) >= 2:
                pos_tags.add(items[1].strip())
                chunk_tags.add(items[2].strip())
        f.close()
        print(f"{len(pos_tags)} POS TAGS: {pos_tags}")
        print(f"{len(chunk_tags)} CHUNK TAGS: {chunk_tags}")