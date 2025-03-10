FROM arm64v8/ros:humble-ros-base-jammy

ENV CXXFLAGS="-std=c++14"

# Install dependencies and Python packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -U empy==3.3.4 pyros-genmsg setuptools==65.5.1

# Clone and build Micro-XRCE-DDS-Agent
WORKDIR /tmp
RUN git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    ldconfig /usr/local/lib/ && \
    cd / && \
    rm -rf /tmp/Micro-XRCE-DDS-Agent

# Install eigen
RUN apt update && apt install -y libeigen3-dev

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install -r /requirements.txt

# Set working directory for the container
WORKDIR /ros2_ws

# Copy the entrypoint script from host to container
COPY entrypoint.sh /entrypoint.sh

# Make entrypoint script executable
RUN chmod +x /entrypoint.sh

# Source ROS environment in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
