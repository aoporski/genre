#!/bin/bash

case "$1" in
  c) TARGET_DIR="finetune_data3/classical" ;;
  h) TARGET_DIR="finetune_data3/hiphop" ;;
  j) TARGET_DIR="finetune_data3/jazz" ;;
  r) TARGET_DIR="finetune_data3/rock" ;;
  *)
    echo "❌ Nieznany skrót klasy '$1'. Użyj: c (classical), h (hiphop), j (jazz), r (rock)"
    exit 1
    ;;
esac

YT_LINK="$2"

if [ -z "$YT_LINK" ]; then
  echo "❌ Podaj link do YouTube jako drugi argument"
  exit 1
fi

mkdir -p "$TARGET_DIR"

yt-dlp -x --audio-format wav -o "$TARGET_DIR/%(title)s.%(ext)s" "$YT_LINK"