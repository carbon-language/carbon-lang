# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

ARG TAG=3.5.6
FROM homebrew/brew:${TAG}

# Install libxcrypt using Homebrew.
RUN brew install libxcrypt --overwrite

# Install all other dependencies using Homebrew.
RUN brew install bazelisk node python@3.9

# Install all python dependencies modules using Pip.
RUN pip3 install -U pip
RUN pip3 install pre-commit black codespell

# Install Clang/LLVM using Homebrew.
# Many Clang/LLVM releases aren't built with options we rely on.
RUN brew install llvm
