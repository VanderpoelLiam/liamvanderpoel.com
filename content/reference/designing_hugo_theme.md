---
title: "Creating a Hugo theme"
date: 2023-10-13T14:50:49+02:00
draft: true
tags:
- hugo
TocOpen: true
ShowToc: true
year: "2023"
month: "2023/10"
---

The goal is to design my own Hugo theme. 

# Clone a starter theme
We will not be able to run `hugo server` without implementing some basic html and css in our theme. For simplicity, I will use the starter theme from [Eric Murphy](https://github.com/ericmurphyxyz/hugo-starter-theme) and make my modifications on top of this. Run:
```
git clone https://github.com/ericmurphyxyz/hugo-starter-theme themes/Solaria
rm -rf themes/Solaria/.git
```
TODO: Add `resources/_gen/images/starter-theme.png` to the site as what the starter theme looks like