#!/usr/bin/env bash
# usage: contact_sheet.sh video.mp4 out.jpg  -> 2x3 grid of frames spread over the clip
FF=${FFMPEG:-ffmpeg}
n=$($FF -i "$1" -map 0:v:0 -c copy -f null - 2>&1 | grep -oE "frame= *[0-9]+" | tail -1 | grep -oE "[0-9]+")
step=$(( n / 6 ))
$FF -loglevel error -y -i "$1" -vf "select='not(mod(n\,$step))',scale=640:-1,tile=3x2" -frames:v 1 -vsync vfr "$2"
