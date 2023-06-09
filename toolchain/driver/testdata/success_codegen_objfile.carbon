// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Currently just makes sure the command doesn't fail. We should add support for testing by disassembling the object file.
//
// RUN: %{carbon} compile --target=x86_64-unknown-linux-gnu --output=test.o %s

fn Main() -> i32 { return 0; }
