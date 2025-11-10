---
title: "Local Websites"
date: 2025-10-07T16:41:12+02:00
draft: false
---

## Motivation

Let's say you have a server that runs some services like Pi-hole, Home Assistant etc. Maybe you also have some internal websites you host. It would be nice if you could access these sites from anywhere and if they had nice domain names like `pihole.internal.kelp.xyz`, but nothing is exposed to the public internet. Our goals are therefore:

1. Remotely access any service (or website) hosted on our server

2. Access these services via human-readable domain names e.g. `pihole.internal.kelp.xyz`

3. Have valid SSL certificates to avoid seeing browser warnings every time you visit your services

4. Do the above without exposing these services to the wider internet

We can do this using [Tailscale](https://tailscale.com/), a local DNS server and a reverse proxy. I assume Tailscale is both installed and running on both the server and your local machine, otherwise follow the [Tailscale quickstart guide](https://tailscale.com/kb/1017/install).

### Prerequisites

We need to know the Tailscale `Machine name` for our server as well as its local ip address in our home network. The machine name can be found under the `Machines` tab in the Tailscale admin console. The local ip address is the first address you see when running `hostname -I` on the server. For the sake of this tutorial assume our server name is `slippery-server` and it has a local ip address of `192.168.0.123`.

We will also need a domain name. In this tutorial, we'll use the domain `kelp.xyz`[^1]. If you don't want to buy a domain name, you can get a free one from [DuckDNS](https://www.duckdns.org/) with the catch being the resulting domain will be quite long e.g. `kelp.duckdns.org`. Otherwise the only restriction is that we need our DNS hosting to support the [DNS-01 challenge](https://letsencrypt.org/docs/challenge-types/#dns-01-challenge). When you buy a domain, by default the registrar (Namecheap, GoDaddy, etc.) also hosts the DNS server. The easiest approach is to therefore buy a domain from a [registrar that supports Let's Encypt DNS-01 validation](https://community.letsencrypt.org/t/dns-providers-who-easily-integrate-with-lets-encrypt-dns-validation/86438). If you already have a domain from a registrar which doesn't support DNS-01 validation (like Namecheap), you can simply move the DNS hosting to somewhere that does by adapting the steps from [Moving DNS hosting from Namecheap to Cloudflare](https://davidisaksson.dev/posts/dns-migration-to-cloudflare/).

[^1]: The motivation for buying a domain is to avoid seeing browser warnings every time you visit your services. If we don't control the domain then we cannot use the DNS-01 challenge to provision SSL certificates. If we didn't care about SSL warnings, we could use pretty much any domain name we want, so long as it doesn't conflict with another public website i.e. don't use `google.com`. There are some special domains to avoid, notably `.local` domains can cause issues down the line (see [Why Using a .local Domain for Internal Networks is a Bad Idea](https://thexcursus.org/why-using-a-local-domain-for-internal-networks-is-a-bad-idea/)).

## Setup a local service

We first need to run some service on our server. Feel free to skip this section if you already have all your services setup. We just need something running for this tutorial, so we pick the dummy webserver [whoami](https://github.com/traefik/whoami) which will run on port `8080`.

### Docker install

We will install our service using Docker, so follow the instructions at [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/) (I'm running Ubuntu on my server, adapt the Docker install accordingly for your OS). Don't forget the [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/) so that non-root users can run docker. Then check everything is installed with `docker run hello-world` which should output:

```shell
>>> docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.
[...]
```

### Start the service

We can then run our dummy service [whoami](https://github.com/traefik/whoami) with:

```shell
docker run -d -p 8080:80 traefik/whoami
```

We now have a webserver that prints OS information and HTTP request to output running on port `8080`. Without any additional setup we can access this site via any machine on out Tailnet via `http://<server-name>:8080/` which in our case is `http://slippery-server:8080/`, or from inside our local network at `http://192.168.0.123:8080/`.

## Human-readable domain names

### Local DNS Server

At this point I want to quickly explain the [Domain Name System (DNS)](https://aws.amazon.com/route53/what-is-dns/) protocol. DNS is how we can type `liamvanderpoel.com` into our browser and be routed to the actual IP address where my website is hosted e.g. `37.16.9.210`. This occurs because after I bought my domain name I went to my DNS provider and created DNS records that publicly store the mapping `liamvanderpoel.com` to `37.16.9.210`. We would like the same thing to occur inside our Tailnet and our local network, where any request ending in `internal.kelp.xyz` gets routed to the `slippery-server` machine. This requires running our own local DNS server.

#### Pi-hole

There are a few options for a local DNS server. If you are already running a [Pi-hole](https://pi-hole.net/), then you already have a local DNS server. Otherwise install [dnsmasq](https://wiki.archlinux.org/title/Dnsmasq). We focus on the configuration with Pi-hole, but as Pi-hole runs dnsmasq under the hood, with either approach we add the following line to our dnsmasq configuration:

```text
address=/internal.kelp.xyz/192.168.0.123
```

where `192.168.0.123` is the local ip address of `slippery-server` machine. In the Pi-hole admin console under `Settings` > `All Settings` > `Misc` enable the setting `misc.etc_dnsmasq_d`. The add the above line to `misc.dnsmasq_lines`. Save and Apply the changes. We now have whenever our Pi-hole gets a request ending in `internal.kelp.xyz` it will route it to the `slippery-server` machine.

Next we need to update the ports used by Pi-hole to avoid conflicts with our reverse proxy. We update the docker compose for Pi-hole to use the `8081` port instead of port `80` for HTTP and then remove the `"443:443/tcp"` line so that the reverse proxy can handle all HTTPS connections:

```yml
ports:
    - "8081:80/tcp"
    # - "443:443/tcp"
```

Then restart the Pi-hole service.

#### Tailscale DNS Settings

Next we need to configure Tailscale to use the DNS server running on the `slippery-server` machine for anything ending in `internal.kelp.xyz`. This requires adding a custom nameserver under the `DNS` tab in the admin console with the ip address of `slippery-server` i.e. `100.764.629.423` to the `Nameserver` field. We now have a decision:

Do we want to use the Pi-hole to block ads whenever we connect to our Tailnet?

* Yes. Then select the `Override DNS servers` option to always use the Pi-hole DNS server when connected to Tailscale.

* No. Then use [Split DNS](https://tailscale.com/learn/why-split-dns) so we only use the Pi-hole DNS server for domains ending in `kelp.xyz`.

We can now test our DNS configuration by running `dig anything.internal.kelp.xyz` on any device in our Tailnet, and we should see the request get routed to the `192.168.0.123` ip address:

```shell
> dig anything.internal.kelp.xyz +short 
192.168.0.123
```

We can also verify that in our browser we can additionally access the service at `http:/internal.kelp.xyz:8080/`.

### Reverse Proxy

So far we can access any service from any machine on our Tailnet using our custom subdomain `internal.kelp.xyz` if we know the port on which each service is running e.g. `whoami` is running on port `8080`, so we type into our browser `http://internal.kelp.xyz:8080/`. Our goal for this section is no longer need port numbers, but to instead access our services by chaining subdomains e.g. `http://whoami.internal.kelp.xyz`. This is the job of our [Reverse proxy](https://en.wikipedia.org/wiki/Reverse_proxy) which routes traffic to the specific services depending on their url.

#### Caddy setup

We will use [Caddy](https://wiki.archlinux.org/title/Caddy) as our reverse proxy. As I will be using Cloudflare as my DNS host, I use a Docker image for a [Caddy server with built-in support for Cloudflare DNS-01 ACME challenges](https://github.com/CaddyBuilds/caddy-cloudflare). Following the setup instructions I add to my existing `docker-compose.yml`:

```yml
services:
    caddy: 
    container_name: caddy 
    image: ghcr.io/caddybuilds/caddy-cloudflare:latest 
    restart: unless-stopped 
    cap_add: 
        - NET_ADMIN 
    ports: 
        - "80:80" 
        - "443:443" 
        - "443:443/udp" 
    volumes: 
        - ../app-data/caddy/Caddyfile:/etc/caddy/Caddyfile 
        - ../app-data/caddy/site:/srv 
        - ../app-data/caddy/data:/data 
        - ../app-data/caddy/config:/config 
    environment: 
        - CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
    network_mode: host
```

replace `your_cloudflare_api_token` with your real Cloudflare API token generated by following the [Creating a Cloudflare API Token](https://github.com/CaddyBuilds/caddy-cloudflare?tab=readme-ov-file#creating-a-cloudflare-api-token) instructions. Ensure you select `Zone:Read` and `DNS:Edit` permissions for the domain(s) you're managing with Caddy.

Then we edit the Caddyfile and add the lines:

```text
{
  # Set the ACME DNS challenge provider to use Cloudflare for all sites
  acme_dns cloudflare {env.CLOUDFLARE_API_TOKEN}
}

whoami.internal.kelp.xyz {
    reverse_proxy http://localhost:8080
}

pihole.internal.kelp.xyz {
    redir / /admin
    reverse_proxy http://localhost:8081
}
```

This will now route `https://whoami.internal.kelp.xyz` to our service on port `8080`. We additionally added an entry for our Pi-hole admin console, with an additional redirect from `https://pihole.internal.kelp.xyz` to `https://pihole.internal.kelp.xyz/admin` as I don't want to have to remember to add `/admin`.

## Local access without Tailscale

Lastly if we are connected to our local network, we want the ability to connect to our services without having to connect to Tailscale. The approach I took was to configure my router to use `slippery-server` as its default DNS server (I also added a backup DNS server like Cloudflare's `1.1.1.1` so that I don't lose my ability to browse the internet if either `slippery-server` or the Pi-hole container goes down). The added benefit is that the Pi-hole will block ads on my entire local network. We can test everything is working by running `dig anything.internal.kelp.xyz` on any device in our local network (with Tailscale disconnected), and we should see the request get routed to the `192.168.0.123` ip address:

```shell
> dig anything.internal.kelp.xyz +short 
192.168.0.123
```

## Conclusion

We can now access our services with nice domain names and no security warnings! Things work both inside our local network and remotely when connected to Tailscale. One decision I made that I think others may not want to follow is running my own local DNS Server. If you want to instead use a public DNS server I can recommend following ideas from [Easy, quick and free valid SSL certificates for your homelab](https://notthebe.ee/blog/easy-ssl-in-homelab-dns01/) and [Remotely access and share your self-hosted services](https://www.youtube.com/watch?v=Vt4PDUXB_fg&t=393s).

{{< katex >}}

{{< reflist exclude="ubuntu.com, nvidia.com, raw.githubusercontent.com, docker.com, turek.dev, community.letsencrypt.org, davidisaksson.dev">}}
