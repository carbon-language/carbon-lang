// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This validates that earlier errors prevent lowering, without crashing.
// AUTOUPDATE

// CHECK:STDERR: toolchain/lowering/testdata/basics/fail_before_lowering.carbon:[[@LINE+1]]:1: Unrecognized declaration introducer.
a;
