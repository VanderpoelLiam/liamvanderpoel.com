---
title: "Local Websites"
date: 2025-10-07T16:41:12+02:00
draft: false
---

## Motivation

Motivation is want a way to

1. Serve websites that we only want accessed inside our Tailnet

2. Give these sites human-readable names

Let's take the example of exposing a Local ChatGPT like interface on our Tailnet, but the idea is the same for any other service or website we might want to host.

## General Prerequisites

Assume already have a server setup. This guide assumes a machine running Ubuntu server with tailscale installed and a NVIDIA GPU configured. See [Tailscale quickstart](https://tailscale.com/kb/1017/install) for details. Next in the Tailscale admin console navigate to the `Machines` tab and note down the machine name for our server e.g. `slippery-server` as well as the ip address e.g. `100.286.448.604`. Then navigate to the `DNS` tab to note down the Tailscale DNS name. I recommend renaming it to a more human readable name e.g. `pompous-pufferfish.ts.net`

## Local LLM Service

Based on [Self-host a local AI stack](https://tailscale.com/blog/self-host-a-local-ai-stack).

### Docker install

We will install our services using Docker, so follow the instructions at [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/). Don't forget the [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/) so that non-root users can run docker. Then check everything is installed with `docker run hello-world` which should output:

```bash
>>> docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.
[...]
```

We also need to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then restart docker with `sudo systemctl restart docker`. Check that the GPU is working with:

```bash
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

### Configuring the docker containers

As a quickstart you can run the [run-compose.sh](https://raw.githubusercontent.com/open-webui/open-webui/refs/heads/main/run-compose.sh) script from Open WebUI:

```bash
git clone git@github.com:open-webui/open-webui.git
cd open-webui/
chmod +x run-compose.sh
./run-compose.sh --enable-gpu
```

If everything goes well you should be able to see the Open WebUI web interface at `http://slippery-server.pompous-pufferfish.ts.net:3000/auth`.

We want a bit more customization, so instead we write a custom `docker-compose.yml`. First create the folder `/opt/services/` for all our services, as it is a shared location accessible to all users. Then create `local-llm/docker-compose.yaml` containing:

START HERE: The custom docker compose using vLLM.

```yaml
TODO: custom docker compose yaml file
```

TODO: tree of `/opt/services/local-llm/` to show full dir structure.


<!-- ## Sidetrack on DNS servers

TODO: What is a DNS server, why do we need our own in this case?

## Setting up our DNS Server

Install [Unbound](https://unbound.docs.nlnetlabs.nl/en/latest/getting-started/installation.html). Check that it is running with `systemctl status unbound`, should see something like:

```bash
>>> systemctl status unbound
● unbound.service - Unbound DNS server
     Loaded: loaded (/usr/lib/systemd/system/unbound.service; enabled; preset: enabled)
     Active: active (running) since Tue 2025-10-07 20:27:19 UTC; 22min ago
[...]
```

Ensure it auto-starts at boot:

```bash
sudo systemctl enable unbound
```

START HERE: How to configure this DNS server!!!

Next we need to add update the configuration by adding a new file `etc/unbound/unbound.conf.d/tailscale.conf`:

```bash
server:
    # specify the interface to answer queries from by ip-address.
    interface: 100.286.448.604

    # only tailnet subnet can connect to the resolver
    access-control: 100.0.0.0/8 allow


    # Respond to DNS queries for your custom domain with your Tailnet IP
    local-zone: "your.domain.com." redirect
    local-data: "your.domain.com. IN A 100.286.448.604"
    local-data-ptr: "100.286.448.604 your.domain.com"

    # send minimal amount of information to upstream servers to enhance privacy
    qname-minimisation: yes
``` -->

{{< katex >}}

{{< reflist >}}
