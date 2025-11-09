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

We can do this using [Tailscale](https://tailscale.com/), a local DNS server and a reverse proxy. I assume Tailscale is both installed and running on both the server and your local machine, otherwise follow the [Tailscale quickstart guide](https://tailscale.com/kb/1017/install). We then need to retrieve the `Tailnet DNS name` and `Machine name` for our server from the Tailscale admin console under the `DNS` and `Machines` tabs respectively. For the sake of this tutorial assume our server name is `slippery-server` and the Tailnet DNS name is `pompous-pufferfish.ts.net`.

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

We now have a webserver that prints OS information and HTTP request to output running on port `8080`. Without any additional setup we can access this site via any machine on out Tailnet via `http://<server-name>:8080/` which in our case is `http://slippery-server:8080/`.

## Human-readable domain names

### Domain Name

In this tutorial, we'll use the domain `kelp.xyz`[^1]. If you don't want to buy a domain name, you can get a free one from [DuckDNS](https://www.duckdns.org/) with the catch being the resulting domain will be quite long e.g. `kelp.duckdns.org`. Otherwise the only restriction is that we need our DNS hosting to support the [DNS-01 challenge](https://letsencrypt.org/docs/challenge-types/#dns-01-challenge). When you buy a domain, by default the registrar (Namecheap, GoDaddy, etc.) also hosts the DNS server so the easiest approach is to buy a domain from any [registrar that supports Let's Encypt DNS-01 validation](https://community.letsencrypt.org/t/dns-providers-who-easily-integrate-with-lets-encrypt-dns-validation/86438). But if you already have a domain from a registrar which doesn't support DNS-01 validation (like Namecheap), you can move the DNS hosting to somewhere that does by following the steps in [Moving DNS hosting from Namecheap to Cloudflare](https://davidisaksson.dev/posts/dns-migration-to-cloudflare/).

[^1]: The motivation for buying a domain is to avoid seeing browser warnings every time you visit your services. If we don't control the domain then we cannot use the DNS-01 challenge to provision SSL certificates. If we didn't care about SSL warnings, we could use pretty much any domain name we want, so long as it doesn't conflict with another public website i.e. don't use `google.com`. There are some special domains to avoid, notably `.local` domains can cause issues down the line (see [Why Using a .local Domain for Internal Networks is a Bad Idea](https://thexcursus.org/why-using-a-local-domain-for-internal-networks-is-a-bad-idea/)).

### Local DNS Server

At this point I want to quickly explain the [Domain Name System (DNS)](https://aws.amazon.com/route53/what-is-dns/) protocol. DNS is how we can type `liamvanderpoel.com` into our browser and be routed to the actual IP address where my website is hosted e.g. `37.16.9.210`. This occurs because after I bought my domain name I went to my DNS provider (e.g. Cloudflare, Namecheap, ...) and created DNS records that publicly store this mapping `liamvanderpoel.com` to `37.16.9.210`. We would like the same thing to occur inside our Tailnet where anything ending in `kelp.xyz` points to the `slippery-server` machine. This requires running our own local DNS server.

#### Pi-hole

There are a few options for a local DNS server. If you are already running a [Pi-hole](https://pi-hole.net/), then you already have a local DNS server. Otherwise install [dnsmasq](https://wiki.archlinux.org/title/Dnsmasq). We focus on the configuration with Pi-hole, but as Pi-hole runs dnsmasq under the hood, with either approach we add the following lines to our DNS configuration:

```text
address=/kelp.xyz/100.764.629.423
```

where `100.764.629.423` is the Tailscale ip address of `slippery-server` which can be found under the `Machines` tab in the Tailscale admin console. In the Pi-hole admin console under `Settings` > `All Settings` > `Misc` enable the setting `misc.etc_dnsmasq_d`. The add the above line to `misc.dnsmasq_lines`. Save and Apply the changes. We now have whenever our Pi-hole gets a request ending in `kelp.xyz` it will route it to the `slippery-server` machine.

Next we need to update the ports used by Pi-hole to avoid conflicts with our reverse proxy. We update the docker compose for Pi-hole to use the `8081` port instead of port `80` for HTTP and then remove the `"443:443/tcp"` line so that the reverse proxy can handle all HTTPS connections:

```yml
ports:
    - "8081:80/tcp"
    # - "443:443/tcp"
```

Then restart the Pi-hole service.

#### Tailscale DNS Settings

Next we need to configure Tailscale to use the DNS server running on the `slippery-server` machine for anything ending in `kelp.xyz`. This requires adding a custom nameserver under the `DNS` tab in the admin console with the ip address of `slippery-server` i.e. `100.764.629.423` to the `Nameserver` field. We now have a decision:

Do we want to use the Pi-hole to block ads whenever we connect to our Tailnet?

* No. Then use [Split DNS](https://tailscale.com/learn/why-split-dns) so we only use the Pi-hole DNS server for domains ending in `kelp.xyz`.

* Yes. Then select the `Override DNS servers` option to always use the Pi-hole DNS server when connected to Tailscale.

We can now test our DNS configuration by running `dig anything.kelp.xyz` on any device in our Tailnet[^2] and we should see the request get routed to the `100.764.629.423` ip address:

[^2]: The `slippery-server` machine is running the Pi-hole service so it does not use it for DNS. Therefore none of this routing will work on that machine.

```shell
> dig anything.kelp.xyz +short 
100.764.629.423
```

We can also verify that in our browser we can additionally access the service at `http:/kelp.xyz:8080/` instead of only at `http://slippery-server:8080/`.

### Reverse Proxy

So far we can access any service from any machine on our Tailnet using our custom domain name `kelp.xyz` if we know the port on which each service is running e.g. `whoami` is running on port `8080`, so we type into our browser `http://vanderpoel.internal:8080/`. Our goal for this section is no longer need port numbers, but instead access our services with subdomains e.g. `http://whoami.kelp.xyz`. This is the job of our [Reverse proxy](https://en.wikipedia.org/wiki/Reverse_proxy) which routes traffic to the specific services depending on their url.

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
```

replace `your_cloudflare_api_token` with your real Cloudflare API token generated by following the [Creating a Cloudflare API Token](https://github.com/CaddyBuilds/caddy-cloudflare?tab=readme-ov-file#creating-a-cloudflare-api-token) instructions. Ensure you select `Zone:Read` and `DNS:Edit` permissions for the domain(s) you're managing with Caddy.

Then we edit the Caddyfile and add the lines:

```text
{
  # Set the ACME DNS challenge provider to use Cloudflare for all sites
  acme_dns cloudflare {env.CLOUDFLARE_API_TOKEN}
}

whoami.kelp.xyz {
    reverse_proxy http://slippery-server:8080
}

pihole.kelp.xyz {
    redir / /admin
    reverse_proxy http://slippery-server:8081
}
```

This will now route `https://whoami.kelp.xyz` to `http://vanderpoel.internal:8080/`. We additionally added an entry for our Pi-hole admin console, with an additional redirect from `https://pihole..kelp.xyz` to `https://pihole.kelp.xyz/admin` as I don't want to have to always remember to add `/admin`.

TODO: Getting a Bad Gateway error.

<!-- -------------------------------------

Now from any machine connected to our Tailnet, typing `https://whoami.kelp.xyz` in our browser should bring up our dummy webserver!

## SSL certificates

At this point we have achieved all of our goals and we could stop here. But one annoying thing is that we get an SSL warning `Warning: Potential Security Risk Ahead` every time we access our internal services. This is because we are using self signed TLS certificates with Caddy. The easiest way rid of these ugly warnings is to buy a domain name and use the [DNS-01 challenge](https://letsencrypt.org/docs/challenge-types/#dns-01-challenge) to get valid SSL certificates.

### Domain Name

If you don't want to buy a domain name, you can get a free one from [DuckDNS](https://www.duckdns.org/) with the catch being the resulting domain will be quite long e.g. `slippery-name.duckdns.org`. Otherwise buy a domain from any [registar that supports Let's Encypt DNS-01 validation](https://community.letsencrypt.org/t/dns-providers-who-easily-integrate-with-lets-encrypt-dns-validation/86438). If you already have a domain from a registar which doesnt support DNS-01 validation e.g. Namecheap, you can move the DNS hosting to somewhere that does by following [Moving DNS hosting from Namecheap to Cloudflare](https://davidisaksson.dev/posts/dns-migration-to-cloudflare/).

TODO: Provide remainder of tutorial based on this https://notthebe.ee/blog/easy-ssl-in-homelab-dns01/ and/or this https://blog.mni.li/posts/internal-tls-with-caddy/?utm_source=shorturl#setting-up-acmesh-and-getting-a-certificate

-------------

Put an entry into Cloudflare as a public DNS record, and setup Caddy to use C

[Configuring Caddy and DNS](https://www.youtube.com/watch?v=Vt4PDUXB_fg&t=214s)

Create a CNAME DNS record that points anything ending in `*.internal.vanderpoel.ch` points to the fully qualified domain name of our server, in our case`slippery-server.pompous-pufferfish.ts.net`

```shell
> dig test.local.vanderpoel.ch +short
slippery-server.pompous-pufferfish.ts.net.
```

Based on the example [Custom domains Caddyfile](https://github.com/tailscale-dev/video-caddy-custom-domains/blob/main/caddy/Caddyfile) we update our Caddyfile to be:

```text
(cloudflare) {
  tls {
    dns cloudflare <CLOUDFLARE-API-TOKEN>
  }
}

whoami.internal.vanderpoel.ch {
    reverse_proxy http://slippery-server:8080
    import cloudflare
}

pihole.internal.vanderpoel.ch {
    redir / /admin
    reverse_proxy http://slippery-server:8081
    import cloudflare
}
```

replace `<CLOUDFLARE-API-TOKEN>` with your real Cloudflare API token generated by following the [Create API token](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/) instructions. Ensure you select `Zone.Zone:Read` and `Zone.DNS:Edit` permissions for the domain(s) you're managing with Caddy. -->

{{< katex >}}

{{< reflist exclude="ubuntu.com, nvidia.com, raw.githubusercontent.com, docker.com, turek.dev">}}
