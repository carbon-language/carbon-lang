// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s 2>&1 | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

// CHECK:{{.*}}/toolchain/parser/testdata/basics/no_intro_without_semi.carbon:[[@LINE+4]]:1: Unrecognized declaration introducer.
// CHECK:[
// CHECK:{node_index: 0, kind: 'FileEnd', text: ''},
// CHECK:]
foo bar baz
