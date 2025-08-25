# ============ STAGE 1: BUILDER (kompiliert Catkin) ============
FROM ros:noetic-ros-base as builder

LABEL org.opencontainers.image.source="https://github.com/christopherjaeger/toh_final"

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
WORKDIR /catkin_ws

# Build- und ROS-Tools für Catkin
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3-pip python3-rosdep python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# rosdep vorbereiten (ohne rosdep init)
RUN rosdep update

# Quellcode reinkopieren
COPY src/ /catkin_ws/src/

# Systemabhängigkeiten deiner Pakete installieren
RUN apt-get update && rosdep install --from-paths src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*

# Catkin-Build mit Debug-Logs (zeigt dir die echte Fehlermeldung)
SHELL ["/bin/bash", "-lc"]
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && catkin config --extend /opt/ros/${ROS_DISTRO} --install \
    && catkin build --summarize --verbose --no-status --cmake-args -DCMAKE_BUILD_TYPE=Release

# ============ STAGE 2: RUNTIME (schlank) ============
FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
WORKDIR /catkin_ws

# Feature-Schalter (true/false) bei Bedarf überschreiben
ARG WITH_GPU=false
ARG WITH_RS=false
ARG WITH_OPEN3D=false
ARG WITH_DEV=false

# Laufzeit-ROS-Pakete: MoveIt, RViz, TF, (optional Gazebo; auskommentieren wenn nicht gebraucht)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-moveit \
    ros-noetic-rviz \
    ros-noetic-tf \
    ros-noetic-tf2-ros \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    # Gazebo nur falls nötig:
    # ros-noetic-gazebo-ros ros-noetic-gazebo-ros-control ros-noetic-ros-control ros-noetic-ros-controllers \
    python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Python Basis
RUN pip3 install --no-cache-dir --upgrade pip

# Schlanke Runtime-Pythonpakete (wähle gezielt nur das, was du wirklich brauchst)
# OpenCV headless spart ~hundert MB ggü. opencv-python
# Ultralytics holt Gewichte beim ersten Start; das Image bleibt trotzdem klein.
RUN pip3 install --no-cache-dir \
    numpy==1.24.4 \
    scipy==1.10.1 \
    transforms3d==0.4.1 \
    pytransform3d==3.14.0 \
    opencv-python-headless==4.11.0.86 \
    pandas==2.0.3 \
    ultralytics==8.3.140

# Optional: PyTorch CPU oder GPU
# CPU: nutzt den CPU-Index und ist deutlich kleiner
# GPU: cu121 Wheels; Container später mit --gpus all starten
RUN if [ "$WITH_GPU" = "true" ]; then \
      pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1; \
    else \
      pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1; \
    fi

# Optional: RealSense (nur wenn du wirklich eine D435/D415 etc. nutzt)
RUN if [ "$WITH_RS" = "true" ]; then \
      pip3 install --no-cache-dir pyrealsense2==2.55.1.6486; \
    fi

# Optional: Open3D (groß – nur wenn wirklich nötig)
RUN if [ "$WITH_OPEN3D" = "true" ]; then \
      pip3 install --no-cache-dir open3d==0.13.0; \
    fi

# Optional: Dev-Umgebung (Jupyter etc.). Für Runtime besser weglassen.
RUN if [ "$WITH_DEV" = "true" ]; then \
      pip3 install --no-cache-dir jupyterlab==3.6.8 ipykernel==6.29.5; \
    fi

# Install-Space aus Builder übernehmen (nur das Gebaute, kein Source, kein Compiler)
COPY --from=builder /catkin_ws/install/ /catkin_ws/install/

# ROS-Umgebung automatisch sourcen
SHELL ["/bin/bash", "-lc"]
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /root/.bashrc \
    && echo 'source /catkin_ws/install/setup.bash' >> /root/.bashrc

# Standard: interaktive Shell; ggf. eigenes Startskript setzen
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]