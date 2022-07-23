# Docker

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->
- ubuntu:20.04
## Build and run image
### Build
Building the base image is required.
```
docker build -t carbon-ubuntu2004-base ./ubuntu2004/base
```
Build image using git repository
```bash
docker build -t carbon ./ubuntu2004/github
```
Build image using copy instruction
```bash
docker build -f ./ubuntu2004/Dockerfile -t carbon ..
```
Run image
```bash
docker run carbon
```
Run image using specific file
```bash
docker run carbon bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```
### Using Volume
```bash
docker run -w /carbon-lang -v /path/to/carbon-lang:/carbon-lang carbon-ubuntu2004-base bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```
-   navigate to repository root
```bash
cd ..
```
-   bash:
```bash
docker run -w /carbon-lang -v $PWD:/carbon-lang carbon bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```
-   cmd: 
```cmd
docker run -w /carbon-lang -v %cd%:/carbon-lang carbon bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```
-   powershell: 
```ps
docker run -w /carbon-lang -v ${PWD}:/carbon-lang carbon bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```
