// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile %s
//
// AUTOUPDATE

// CHECK:STDERR: fail_multiline_token.carbon:[[@LINE+6]]:1: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: '''
// CHECK:STDERR: ^~~
// CHECK:STDERR: fail_multiline_token.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `HandleInvalidParse`.
// CHECK:STDERR: '''
// CHECK:STDERR: ^~~
'''
hello
'''
