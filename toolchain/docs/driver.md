# Driver

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)

<!-- tocstop -->

## Overview

The driver provides commands and ties together the toolchain's flow. Running a
command such as `carbon compile --phase=lower <file>` will run through the flow
and print output. Several dump flags, such as `--dump-parse-tree`, print output
in YAML format for easier parsing.
