# Docker

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

- ubuntu:20.04

## Base image
```bash
docker build -f Base.Dockerfile --tag carbon-ubuntu2004-base .
```
## Git based example image
```bash
docker build --tag carbon-example .
```
```bash
docker run carbon-example
```
## Use base image with volume

```bash
docker run -w /carbon-lang -v /path/to/carbon-lang:/carbon-lang carbon-ubuntu2004-base bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```
-   navigate to repository root
```bash
cd ..
```
-   bash:
```bash
docker run -w /carbon-lang -v $PWD:/carbon-lang carbon-ubuntu2004-base bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```
-   cmd: 
```cmd
docker run -w /carbon-lang -v %cd%:/carbon-lang carbon-ubuntu2004-base bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```
-   powershell: 
```ps
docker run -w /carbon-lang -v ${PWD}:/carbon-lang carbon-ubuntu2004-base bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```