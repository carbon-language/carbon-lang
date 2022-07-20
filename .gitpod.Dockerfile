# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gitpod/workspace-full

USER gitpod

# install and configure llvm/bazelisk
RUN brew update && brew install bazelisk llvm pre-commit
RUN echo 'export PATH="$(brew --prefix llvm)/bin:${PATH}"' >> /home/gitpod/.bashrc

# store the blaze cache in a place where it will be persisted
RUN mkdir -p /home/gitpod/.cache/bazel && ln -s /home/gitpod/.cache/bazel /workspace/.bazel
