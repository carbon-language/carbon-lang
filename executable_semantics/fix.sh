#!/bin/bash -eux
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

sed -i s/$1/$2/ *.h *.cpp *.ypp
