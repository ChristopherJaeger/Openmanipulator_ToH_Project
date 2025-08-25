## Pull
docker pull ghcr.io/christopherjaeger/toh_final:latest

## Run (CPU)
docker run --rm -it ghcr.io/christopherjaeger/toh_final:latest

## Run (GPU, wenn du eine GPU-Variante baust)
docker run --rm -it --gpus all ghcr.io/christopherjaeger/toh_final:runtime-gpu

## X11 (RViz/Gazebo auf Linux-Host)
xhost +local:root
docker run --rm -it \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  ghcr.io/christopherjaeger/toh_final:latest
