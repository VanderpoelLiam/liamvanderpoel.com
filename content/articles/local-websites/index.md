---
title: "Local Websites"
date: 2025-10-07T16:41:12+02:00
draft: false
---

## Motivation

Let's say you have an ubuntu home server that runs some services like Immich, Home Assistant etc. Maybe you also have some internal websites you host. It would be nice if you could access these sites remotely and if they had nice domain names like `immich.vanderpoel.local`, but we don't want expose anything to the public internet. Our goals are therefore:

1. Remotely access any service hosted on our server

2. Access these services via human-readable domain names e.g. `immich.vanderpoel.local`

3. Do this without exposing these services to the wider internet

We can do this using [Tailscale](https://tailscale.com/), [Split DNS](https://tailscale.com/learn/why-split-dns) a local DNS server and a reverse proxy. I assume Tailscale is already installed on both the server and your local machine, otherwise follow the [Tailscale quickstart guide](https://tailscale.com/kb/1017/install). We also need to retrieve the `Tailnet DNS name` and `Machine name` for our server from the Tailscale admin console under the `DNS` and `Machines` tabs respectively. For the sake of this tutorial assume our server name is `slippery-server` and the Tailnet DNS name is `pompous-pufferfish.ts.net`.

## Setup a local service

We need to first run some service on our ubuntu home server. Let's take the example of running our own ChatGPT like interface based on [Self-host a local AI stack](https://tailscale.com/blog/self-host-a-local-ai-stack). Feel free to skip this section if you already know how to setup a service on your home server. The important point is that when we are done we will have a local LLM service running on `http://localhost:3000/` that is also accessible at `http://slippery-server.pompous-pufferfish.ts.net:3000/` from any device in our Tailnet.

### NVIDIA GPU setup

Follow the [Ubuntu Server documentation](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/index.html) to install the necessary drivers. We also need to install CUDA per [the NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/contents.html) (we selected the proprietary kernel module flavor). An important post-installation step is:

```shell
sudo tee /etc/profile.d/cuda.sh > /dev/null <<'EOF'
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
EOF

sudo chmod +x /etc/profile.d/cuda.sh
```

### Docker install

We will install our services using Docker, so follow the instructions at [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/). Don't forget the [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/) so that non-root users can run docker. Then check everything is installed with `docker run hello-world` which should output:

```shell
>>> docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.
[...]
```

We also need to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then restart docker with `sudo systemctl restart docker`. You can check that the GPU is working inside docker with:

```shell
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

### Running a Local ChatGPT Interface

For the purposes of this tutorial, you can use the [run-compose.sh](https://raw.githubusercontent.com/open-webui/open-webui/refs/heads/main/run-compose.sh) script from Open WebUI:

```shell
git clone git@github.com:open-webui/open-webui.git
cd open-webui/
chmod +x run-compose.sh
./run-compose.sh --enable-gpu
```

We now have Open WebUI running on `http://localhost:3000/` which can be accessed via our local machine with Tailscale running at `http://<server-name>.<tailnet-name>.ts.net:3000/` which in our case is `http://slippery-server.pompous-pufferfish.ts.net:3000/`.

## Human-readable domain names

We have now achieved our first goal which is to have remote access any service hosted on `slippery-server`. In our case we have Open WebUI running at `http://slippery-server.pompous-pufferfish.ts.net:3000/`. The next goal is have this site accessible instead at `https://immich.vanderpoel.local`.

TODO: START HERE - how are we going to have human readable domain names and https

<!-- ### HTTP to HTTPS

Right now nothing is encrypted and we have to remember what port we are serving. To fix both these issues, navigate to the `DNS` tab in the admin console and turn on both `MagicDNS` and `HTTPS Certificates`. This allows us to use [Tailscale Serve](https://tailscale.com/kb/1312/serve):

```shell
tailscale serve --bg 3000
```

and now our service will be made available at `https://slippery-server.pompous-pufferfish.ts.net`.

Note: This does put `http://slippery-server.pompous-pufferfish.ts.net` in the public ledger, so it is important to state **Do not enable the HTTPS feature if any of your machine names contain sensitive information.** See [Enabling HTTPS](https://tailscale.com/kb/1153/enabling-https) for more details.


## Pretty names

START HERE: How to have custom domain name for this service so can have `tailscale serve` serve multiple services on the same machine all with nice custom names


The [Domain Name System (DNS)](https://aws.amazon.com/route53/what-is-dns/) protocol is how we can type `liamvanderpoel.com` into our browser instead of its actual IP address like `37.16.9.210`. We would like to have the same thing occur for our local services inside our Tailnet, this is where [Split DNS](https://tailscale.com/learn/why-split-dns) comes in. If we are connected to Tailscale and have our own DNS server setup *inside* the tailnet, then we can have it so that anything ending in `.internal.liamvanderpoel.com` gets routed to our internal services.

## Setting up own DNS Server

Install [Unbound](https://unbound.docs.nlnetlabs.nl/en/latest/getting-started/installation.html). Check that it is running with `systemctl status unbound`, should see something like:

```shell
>>> systemctl status unbound
● unbound.service - Unbound DNS server
     Loaded: loaded (/usr/lib/systemd/system/unbound.service; enabled; preset: enabled)
     Active: active (running) since Tue 2025-10-07 20:27:19 UTC; 22min ago
[...]
```

Ensure it auto-starts at boot:

```shell
sudo systemctl enable unbound
```

START HERE: How to configure this DNS server!!!

Next we need to add update the configuration by adding a new file `/etc/unbound/unbound.conf.d/tailscale.conf`:

TODO: Anonymize this data after get it working

```shell
server:
    # Listen to queries from the Tailnet
    interface: tailscale0

    # Answer queries from the Tailnet
    access-control: 100.0.0.0/8 allow


    # Respond to DNS queries for your custom domain with your Tailnet IP
    local-zone: "pompous-pufferfish.com." redirect
    local-data: "pompous-pufferfish.com. IN A server.pompous-pufferfish.ts.net"
    local-data-ptr: "server.pompous-pufferfish.ts.net pompous-pufferfish.com"

    # send minimal amount of information to upstream servers to enhance privacy
    qname-minimisation: yes
```

See the [unbound.conf](https://unbound.docs.nlnetlabs.nl/en/latest/manpages/unbound.conf.html) man page for more details. -->

{{< katex >}}

{{< reflist >}}
