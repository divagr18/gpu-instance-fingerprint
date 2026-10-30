# Probe image: CUDA runtime + numpy + cupy + sshd, with probes/probe_puf.py baked
# in, so a rented GPU machine boots ready to run the active probe.
# Build from the repo root:
#   docker build -f docker/Dockerfile -t gpu-instance-fingerprint .
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip openssh-server ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh

RUN pip install --break-system-packages --no-cache-dir numpy cupy-cuda12x

COPY probes/probe_puf.py /root/probe_puf.py
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

WORKDIR /root
EXPOSE 22
CMD ["/start.sh"]
