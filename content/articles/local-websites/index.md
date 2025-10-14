---
title: "Local Websites"
date: 2025-10-07T16:41:12+02:00
draft: false
---

## Motivation

Let's say you have an ubuntu home server that runs some services like Immich, Home Assistant etc. Maybe you also have some internal websites you host. It would be nice if you could access these sites remotely and if they had nice domain names like `immich.vanderpoel.internal`, but we don't want expose anything to the public internet. Our goals are therefore:

1. Remotely access any service hosted on our server

2. Access these services via human-readable domain names e.g. `immich.vanderpoel.internal`

3. Do this without exposing these services to the wider internet

We can do this using [Tailscale](https://tailscale.com/), [Split DNS](https://tailscale.com/learn/why-split-dns) a local DNS server and a reverse proxy. I assume Tailscale is already installed on both the server and your local machine, otherwise follow the [Tailscale quickstart guide](https://tailscale.com/kb/1017/install). We also need to retrieve the `Tailnet DNS name` and `Machine name` for our server from the Tailscale admin console under the `DNS` and `Machines` tabs respectively. For the sake of this tutorial assume our server name is `slippery-server` and the Tailnet DNS name is `pompous-pufferfish.ts.net`.

## Setup a local service

We need to first run some service on our ubuntu home server. Feel free to skip this section if you already have a service setup on your home server. The important point is that when we are done we will have a service running on `http://localhost:8080/` that is also accessible at `http://slippery-server.pompous-pufferfish.ts.net:8080/` from any device in our Tailnet.

### Docker install

We will install our services using Docker, so follow the instructions at [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/). Don't forget the [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/) so that non-root users can run docker. Then check everything is installed with `docker run hello-world` which should output:

```shell
>>> docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.
[...]
```

We can then run our dummy service [whoami](https://github.com/traefik/whoami) with:

```shell
docker run -d -p 8080:80 traefik/whoami
```

We now have a webserver that prints OS information and HTTP request to output running on `http://localhost:8080/` which can be accessed via our local machine with Tailscale running at `http://<server-name>.<tailnet-name>.ts.net:8080/` which in our case is `http://slippery-server.pompous-pufferfish.ts.net:8080/`.

## Human-readable domain names

### Local DNS Server

The [Domain Name System (DNS)](https://aws.amazon.com/route53/what-is-dns/) protocol is how we can type `liamvanderpoel.com` into our browser and be routed to the actual IP address where my website is hosted e.g. `37.16.9.210`. This occurs because after I bought my domain name I went to my DNS provider (e.g. Cloudflare, Namecheap, ...) and created DNS records that publicly store this mapping `liamvanderpoel.com` to `37.16.9.210` (i.e. the A, AAAA, CNAME records). We would like the same thing to occur inside our Tailnet where anything ending in `vanderpoel.internal` points to the `slippery-server` machine.

Recall our goal is to access our local services hosted on the `slippery-server` machine. We certainly can just buy another domain, and point it to the internal Tailscale ip address of `slippery-server`. But as we only need this DNS resolution to work locally inside our Tailnet, we can instead run our own DNS server who's job is to map a domain of our choosing to the Tailscale ip address of `slippery-server`. We can use any domain name we want[^1], but in practice it make sense not to use a domain already in use.

[^1]: This is not quite right, there are actually some restrictions. In my first draft of this article I used `vanderpoel.local` instead of `vanderpoel.internal` but read that `.local` domains can cause issues down the line (see [Why Using a .local Domain for Internal Networks is a Bad Idea](https://thexcursus.org/why-using-a-local-domain-for-internal-networks-is-a-bad-idea/)).

We therefore install [dnsmasq](https://wiki.archlinux.org/title/Dnsmasq) on `slippery-server`[^2]. A pre-installation step however is to disable the stub DNS server run by systemd-resolved, otherwise you run into the error `failed to create listening socket for port 53: Address already in use`. Following [this guide](https://www.turek.dev/posts/disable-systemd-resolved-cleanly/) we run:

```shell
sudo mkdir -p /etc/systemd/resolved.conf.d/
echo -e "[Resolve]\nDNSStubListener=no" | sudo tee /etc/systemd/resolved.conf.d/disable-stub.conf
sudo ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf
sudo systemctl restart systemd-resolved
```

[^2]: The local DNS server can be installed on any machine inside the Tailnet, it should just be a machine that is always running.


You should now not see anything running on port `53` with `sudo lsof -i :53`. We can now install [dnsmasq](https://wiki.archlinux.org/title/Dnsmasq):

```shell
sudo apt install dnsmasq
```

Then run `sudo vim /etc/dnsmasq.conf` and add the lines:

```text
# Local domain
address=/vanderpoel.internal/100.764.629.423
```

The ip address `100.764.629.423` is that of `slippery-server.pompous-pufferfish.ts.net` and can be found under the `Machines` tab in the admin console.

Finally restart dnsmasq:

```shell
sudo systemctl restart dnsmasq
```

If you get the error `Failed to set DNS configuration: Link lo is loopback device.` then edit the file `/etc/default/dnsmasq` and uncomment the lines:

```shell
IGNORE_RESOLVCONF=yes
DNSMASQ_EXCEPT="lo"
```

and restarting dnsmasq should resolve the issue.

TODO: This does not work do not get expected output:

```shell
> dig anything.vanderpoel.internal
[...]

;; ANSWER SECTION:
anything.vanderpoel.internal. 0   IN      A       100.764.629.423
[...]
```

### Reverse Proxy

TODO: What is a reverse proxy and why is it needed.


<!-- We have now achieved our first goal which is to have remote access to any service hosted on `slippery-server`. In our case we have Open WebUI running at `http://slippery-server.pompous-pufferfish.ts.net:3000/`. The next goal is have this site accessible instead at `https://immich.vanderpoel.internal`.

### Local DNS Server

The [Domain Name System (DNS)](https://aws.amazon.com/route53/what-is-dns/) protocol is how we can type `liamvanderpoel.com` into our browser and be routed to the actual IP address where my website is hosted e.g. `37.16.9.210`. This occurs because after I bought my domain name I went to my DNS provider (e.g. Cloudflare, Namecheap, ...) and created DNS records that publicly store this mapping `liamvanderpoel.com` to `37.16.9.210` (i.e. the A, AAAA, CNAME records). We would like the same thing to occur inside our Tailnet where anything ending in `vanderpoel.internal` points to the `slippery-server` machine.

Recall our goal is to access our local services hosted on the `slippery-server` machine. This machine doesn't have a publicly reachable ip address, so we can't just point a domain to our server like I did for my website. Additionally we only need this mapping to work inside our Tailnet, so the solution is to run our own local DNS server. Its only job is to map a domain of our choosing to the private ip address of `slippery-server` inside the Tailnet. We can use any domain name we want[^1], but in practice it make sense not to use a domain already in use.

[^1]: This is not quite right, there are actually some restrictions. In my first draft of this article I used `vanderpoel.local` instead of `vanderpoel.internal` but read that `.local` domains can cause issues down the line (see [Why Using a .local Domain for Internal Networks is a Bad Idea](https://thexcursus.org/why-using-a-local-domain-for-internal-networks-is-a-bad-idea/)).

We therefore install [dnsmasq](https://wiki.archlinux.org/title/Dnsmasq) on `slippery-server`[^2]:

[^2]: The local DNS server can be installed on any machine inside the Tailnet, it should just be a machine that is always running.

```shell
sudo apt install dnsmasq
```

Then run `sudo vim /etc/dnsmasq.conf` and add the lines:

```text
# Only bind to Tailscale interface
interface=tailscale0
bind-dynamic

# Local domain
address=/vanderpoel.internal/100.764.629.423
```

The ip address `100.764.629.423` is that of `slippery-server.pompous-pufferfish.ts.net` and can be found under the `Machines` tab in the admin console.

Finally restart dnsmasq:

```shell
sudo systemctl restart dnsmasq
```

We then need to setup [Split DNS](https://tailscale.com/learn/why-split-dns) to route anything ending in `vanderpoel.internal` to the `slippery-server` machine. This requires adding a custom nameserver under the `DNS` tab in the admin console. Add the ip address of `slippery-server` i.e. `100.764.629.423` to the `Nameserver` field and add the domain to the `Domain` field i.e. `vanderpoel.internal`. It should look like:

![Split DNS setup in Tailscale admin console](split-dns.png)

We can now test our dns by running `dig anything.vanderpoel.internal` on any device in our Tailnet and we should see the request get routed to the `100.764.629.423` ip address:

```shell
> dig anything.vanderpoel.internal
[...]

;; ANSWER SECTION:
anything.vanderpoel.internal. 0   IN      A       100.764.629.423
[...]
```

### Reverse Proxy

TODO: What is a reverse proxy and why is it needed.

First install [caddy](https://wiki.archlinux.org/title/Caddy):

```shell
sudo apt install caddy
```

We now want to route `https://chat.vanderpoel.internal` to `http://localhost:3000/`. So we edit the caddy file with `sudo vim /etc/caddy/Caddyfile` and add the lines:

```text
chat.vanderpoel.internal {
    reverse_proxy slippery-server.pompous-pufferfish.ts.net:3000
    tls internal
}
```

Lastly restart caddy:

```shell
sudo systemctl restart caddy
```

Now from any machine connected to our Tailnet, typing `https://chat.vanderpoel.internal` in our browser should bring us the Open WebUI web interface!


TODO: immich.vanderpoel.internal pointing to the immich service is due to reverse proxy like caddy, not due to DNS setup -->

<!-- ### HTTP to HTTPS

Right now nothing is encrypted and we have to remember what port we are serving. To fix both these issues, navigate to the `DNS` tab in the admin console and turn on both `MagicDNS` and `HTTPS Certificates`. This allows us to use [Tailscale Serve](https://tailscale.com/kb/1312/serve):

```shell
tailscale serve --bg 3000
```

and now our service will be made available at `https://slippery-server.pompous-pufferfish.ts.net`.

Note: This does put `http://slippery-server.pompous-pufferfish.ts.net` in the public ledger, so it is important to state **Do not enable the HTTPS feature if any of your machine names contain sensitive information.** See [Enabling HTTPS](https://tailscale.com/kb/1153/enabling-https) for more details.
-->

{{< katex >}}

{{< reflist exclude="ubuntu.com, nvidia.com, raw.githubusercontent.com, docker.com, turek.dev">}}
