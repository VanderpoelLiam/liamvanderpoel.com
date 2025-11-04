---
title: "Local Websites"
date: 2025-10-07T16:41:12+02:00
draft: false
---

## Motivation

Let's say you have a server that runs some services like Immich, Home Assistant etc. Maybe you also have some internal websites you host. It would be nice if you could access these sites remotely and if they had nice domain names like `immich.vanderpoel.internal`, but we don't want expose anything to the public internet. Our goals are therefore:

1. Remotely access any service (or website) hosted on our server

2. Access these services via human-readable domain names e.g. `immich.vanderpoel.internal`

3. Do this without exposing these services to the wider internet

We can do this using [Tailscale](https://tailscale.com/), [Split DNS](https://tailscale.com/learn/why-split-dns), a local DNS server and a reverse proxy. I assume Tailscale is already installed on both the server and your local machine, otherwise follow the [Tailscale quickstart guide](https://tailscale.com/kb/1017/install). We then need to retrieve the `Machine name` for our server from the Tailscale admin console under the `Machines` tab. For the sake of this tutorial assume our server name is `slippery-server`.

## Setup a local service

We first need to run some service on our server. Feel free to skip this section if you already have your all your services setup. We just need some service running for this tutorial, so I picked the dummy webserver [whoami](https://github.com/traefik/whoami) which will run on port `8080`. 

### Docker install

We will install our service using Docker, so follow the instructions at [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/). Don't forget the [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/) so that non-root users can run docker. Then check everything is installed with `docker run hello-world` which should output:

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

We now have a webserver that prints OS information and HTTP request to output running on port `8080`. Without any additional setup we can access this site via any machine on out Tailnet via `http://<server-name>:8080/` which in our case is `http://slippery-server:8080/`.

## Human-readable domain names

### Local DNS Server

The [Domain Name System (DNS)](https://aws.amazon.com/route53/what-is-dns/) protocol is how we can type `liamvanderpoel.com` into our browser and be routed to the actual IP address where my website is hosted e.g. `37.16.9.210`. This occurs because after I bought my domain name I went to my DNS provider (e.g. Cloudflare, Namecheap, ...) and created DNS records that publicly store this mapping `liamvanderpoel.com` to `37.16.9.210` (i.e. the A, AAAA, CNAME records). We would like the same thing to occur inside our Tailnet where anything ending in `vanderpoel.internal` points to the `slippery-server` machine.

Recall our goal is to access our local services hosted on the `slippery-server` machine. We certainly can just buy another domain, and point it to the internal Tailscale ip address of `slippery-server`. But as we only need this DNS resolution to work locally inside our Tailnet, we can instead run our own DNS server who's job is to map a domain of our choosing to the Tailscale ip address of `slippery-server`. We can use any domain name we want[^1], but in practice it make sense not to use a domain already in use.

[^1]: This is not quite right, there are actually some restrictions. In my first draft of this article I used `vanderpoel.local` instead of `vanderpoel.internal` but read that `.local` domains can cause issues down the line (see [Why Using a .local Domain for Internal Networks is a Bad Idea](https://thexcursus.org/why-using-a-local-domain-for-internal-networks-is-a-bad-idea/)).

#### Pi-hole

TODO: Wildcard DNS pihole https://www.youtube.com/watch?v=Uzcs97XcxiE&t=1233s

There are a few options for a local DNS server. However if you are already running a [Pi-hole](https://pi-hole.net/), you already have a local DNS server. In the Pi-hole admin console under `Local DNS Records` add `vanderpoel.internal` as a domain pointing to the Tailscale ip address of the `slippery-server` machine e.g. `100.764.629.423`. This can be found under the `Machines` tab in the Tailscale admin console.

We then need to setup [Split DNS](https://tailscale.com/learn/why-split-dns) to route anything ending in `vanderpoel.internal` to the `slippery-server` machine. This requires adding a custom nameserver under the `DNS` tab in the admin console. Add the ip address of `slippery-server` i.e. `100.764.629.423` to the `Nameserver` field and add the domain to the `Domain` field i.e. `vanderpoel.internal`. It should look like:

![Split DNS setup in Tailscale admin console](split-dns.png)

If you instead want all domains to use the Pi-hole DNS resolver when connected to Tailscale, then unclick the `Restrict to domain` option and select the `Override DNS servers` option after you save the configuration. We can now test our DNS by running `dig vanderpoel.internal` on any device in our Tailnet[^2] and we should see the request get routed to the `100.764.629.423` ip address:

[^2]: The `slippery-server` machine is running the Pi-hole service so it does not use it for DNS. Therefore none of this routing will work on that machine.

```shell
> dig vanderpoel.internal
[...]

;; ANSWER SECTION:
vanderpoel.internal. 0   IN      A       100.764.629.423
[...]
```

We can also verify that in our browser we can additionally access the service at `http://vanderpoel.internal:8080/` instead of only at `http://slippery-server:8080/`.

### Reverse Proxy

So far we can access any service from any machine on our Tailnet using our custom domain name `vanderpoel.internal` if we know what port the service is running on e.g. `whoami` is running on port `8080`, so we type into our browser `http://vanderpoel.internal:8080/`. Our goal for this section is to have `http://whoami.vanderpoel.internal` route to the service at port `8080`. This is the job of our [Reverse proxy](https://en.wikipedia.org/wiki/Reverse_proxy) which routes traffic to the specific services depending on their url e.g. `http://whoami.vanderpoel.internal` routes to `http://vanderpoel.internal:8080/`, `http://immich.vanderpoel.internal` routes to `http://vanderpoel.internal:8081/`, etc.

#### Pi-hole setup

First we need to update the ports used by Pi-hole to avoid conflicts with [Caddy](https://wiki.archlinux.org/title/Caddy), the reverse proxy we are going to use. We update the docker compose for Pi-hole to use the `8081` port instead of port `80` for http and then remove the `"443:443/tcp"` line so that Caddy can handle all HTTPS connections:

```yml
ports:
    - "8081:80/tcp"
    # - "443:443/tcp"
```

Next we have a choice about our DNS settings for custom urls:

1. We need to add CNAME records for each of our services in the Pi-hole `Local DNS Settings` e.g. add `whoami.vanderpoel.internal` as a Domain with Target `vanderpoel.internal` for each service.

2. We setup [DNS Wildcards](https://www.youtube.com/watch?v=Uzcs97XcxiE&t=990s) so that anything ending in `vanderpoel.internal` gets routed to the `slippery-server` machine.

Option 1. is the simplest approach, but I didn't want to have to update my Pi-hole DNS setting every time I add a new service. So to setup option 2. we start by deleting our local DNS records as they are no longer needed. We then enable `Should FTL load additional dnsmasq configuration files from /etc/dnsmasq.d/?` under `Settings` > `All Settings` > `Misc` > `misc.etc_dnsmasq_d`. Then add the following line to `misc.dnsmasq_lines`:

```text
address=/vanderpoel.internal/100.764.629.423
```

where `100.764.629.423` is the ip address of `slippery-server` and Save and Apply the changes. We should now have that anything ending in `vanderpoel.internal` resolves to the `slippery-server` machine:

```shell
> dig anything.vanderpoel.internal
[...]

;; ANSWER SECTION:
anything.vanderpoel.internal. 0   IN      A       100.764.629.423
[...]
```

#### Caddy setup

Now we install [Caddy](https://wiki.archlinux.org/title/Caddy) our reverse proxy:

```shell
sudo apt install caddy
```

To route `https://whoami.vanderpoel.internal` to `http://vanderpoel.internal:8080/` we edit the caddy file with `sudo vim /etc/caddy/Caddyfile` and add the lines:

```text
whoami.vanderpoel.internal {
    reverse_proxy localhost:8080
    tls internal
}
```

We additionally add an entry for our Pi-hole admin console, with an additional redirect from `https://pihole.vanderpoel.internal` to `https://pihole.vanderpoel.internal/admin` as I kept forgetting to add the `/admin`:

```text
pihole.vanderpoel.internal {
    redir / /admin
    reverse_proxy localhost:8081
    tls internal
}
```

We lastly restart caddy:

```shell
sudo systemctl restart caddy
```

Now from any machine connected to our Tailnet, typing `https://whoami.vanderpoel.internal` in our browser should bring up our dummy webserver!


{{< katex >}}

{{< reflist exclude="ubuntu.com, nvidia.com, raw.githubusercontent.com, docker.com, turek.dev">}}
