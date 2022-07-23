# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM carbon-ubuntu2004-base as runner

RUN git clone https://github.com/carbon-language/carbon-lang

WORKDIR /carbon-lang
RUN pre-commit install
RUN bazel build //explorer

CMD ["bazel", "run", "//explorer", "--", "./explorer/testdata/print/format_only.carbon"]