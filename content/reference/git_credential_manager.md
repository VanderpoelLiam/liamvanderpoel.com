---
title: "Git Credential Manager"
date: 2023-10-03T19:58:56+02:00
draft: false
tags:
- git
TocOpen: true
ShowToc: true
year: "2023"
month: "2023/10"
---

## What problem does this address?
I want a simple way to push/pull code from GitHub without having to enter my password or username.

## Setup
The installation instructions are adapted from those in the [git credential manager repository](https://github.com/git-ecosystem/git-credential-manager). 

### Download git-credential-manager

Download the latest .deb package, and run the following:
```
sudo dpkg -i <path-to-package>
git-credential-manager configure
```

### Select credential store
There were a few options for how to store credentials. I went with [the GPG/pass approach](https://github.com/git-ecosystem/git-credential-manager/blob/main/docs/credstores.md). To select this store:
```
git config --global credential.credentialStore gpg
```

We then need to install the required packages. GPG is installed by default in most distributions, `pass` was not installed on mine so run:
```
sudo apt install pass
```

Next generate the GPG key pair with:
```
gpg --gen-key
```

Following the prompts will give an output of the form:
```
❯ gpg --gen-key                                                   
gpg (GnuPG) 2.2.27; Copyright (C) 2021 Free Software Foundation, Inc.
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Note: Use "gpg --full-generate-key" for a full featured key generation dialog.

GnuPG needs to construct a user ID to identify your key.

Real name: fake name
Email address: fake_name@email.com
You selected this USER-ID:
    "fake name <fake_name@email.com>"

Change (N)ame, (E)mail, or (O)kay/(Q)uit? o
[...]
pub   rsa3072 2023-10-03 [SC] [expires: 2025-10-02]
      <gpg-id>
uid                      fake name <fake_name@email.com>
sub   rsa3072 2023-10-03 [E] [expires: 2025-10-02]
```
where `<gpg-id>` is a big hexadecimal number of the form `1B61835F2E64E1C62A9A472`. We need this `<gpg-id>` to initalize the store with:
```
pass init <gpg-id>
```

Now next time we run `git push`, it should prompt a sign-in with Github.
