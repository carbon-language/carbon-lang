# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM carbon-ubuntu2404-base as carbon-ubuntu2404-build

COPY . /carbon-lang
WORKDIR /carbon-lang

# Build
RUN bazel build //explorer

FROM carbon-ubuntu2404-build

CMD ["bazel", "run", "//explorer", "--", "./explorer/testdata/print/format_only.carbon"]
