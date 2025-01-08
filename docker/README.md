# Docker

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

ubuntu:22.04

## Build and run image

### Build

Building the base image is required.

```
docker build -t carbon-ubuntu2404-base ./ubuntu2404/base
```

Build image using git repository

```bash
docker build -t carbon-ubuntu2404 ./ubuntu2404/github
```

Build image using copy instruction

```bash
docker build -f ./ubuntu2404/Dockerfile -t carbon-ubuntu2404 ..
```

Run image

```bash
docker run carbon-ubuntu2404
```

Run image using specific file

```bash
docker run carbon-ubuntu2404 bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```

### Using a mounted volume

Run from the repository root for PWD.

```
cd ..
```

```bash
docker run -w "/carbon-lang" -v "${PWD}:/carbon-lang" "carbon-ubuntu2404-base" bazel run "//explorer" -- "./explorer/testdata/print/format_only.carbon"
```
