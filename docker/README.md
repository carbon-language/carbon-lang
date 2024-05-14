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
docker build -t carbon-ubuntu2204-base ./ubuntu2204/base
```

Build image using git repository

```bash
docker build -t carbon-ubuntu2204 ./ubuntu2204/github
```

Build image using copy instruction

```bash
docker build -f ./ubuntu2204/Dockerfile -t carbon-ubuntu2204 ..
```

Run image

```bash
docker run carbon-ubuntu2204
```

Run image using specific file

```bash
docker run carbon-ubuntu2204 bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```

### Using a mounted volume

Run from the repository root for PWD.

```
cd ..
```

```bash
docker run -w "/carbon-lang" -v "${PWD}:/carbon-lang" "carbon-ubuntu2204-base" bazel run "//explorer" -- "./explorer/testdata/print/format_only.carbon"
```
