#!/bin/bash

case "$1" in
  c) TARGET_DIR="additional_data/classical" ;;
  h) TARGET_DIR="additional_data/hiphop" ;;
  j) TARGET_DIR="additional_data/jazz" ;;
  r) TARGET_DIR="additional_data/rock" ;;
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