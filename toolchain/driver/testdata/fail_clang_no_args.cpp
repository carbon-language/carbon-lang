// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: clang --
//
// SET-CAPTURE-CONSOLE-OUTPUT
// clang-format off
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_clang_no_args.cpp
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_clang_no_args.cpp
// CHECK:STDERR: From: Core.IntLiteral To: UInt(32)
// CHECK:STDERR: From: Core.IntLiteral To: Int(32)
// CHECK:STDERR: From: Core.IntLiteral To: Int(32)
// CHECK:STDERR: From: From: Core.IntLiteralCore.IntLiteral To:  To: UInt(32)UInt(32)
// CHECK:STDERR:
// CHECK:STDERR: From: From: Core.IntLiteralCore.IntLiteral To:  To: Int(32)Int(32)
// CHECK:STDERR:
// CHECK:STDERR: From: Core.IntLiteral To: Int(32)
// CHECK:STDERR: From: From: Core.IntLiteralCore.IntLiteral To:  To: UInt(32)From: UInt(32)
// CHECK:STDERR: Core.IntLiteral
// CHECK:STDERR:  To: Int(32)
// CHECK:STDERR: From: Core.IntLiteralFrom:  To: Core.IntLiteral To: UInt(32)Int(32)
// CHECK:STDERR:
// CHECK:STDERR: From: Core.IntLiteral To: i32
// CHECK:STDERR: From: Core.IntLiteral To: Int(32)
// CHECK:STDERR: From: Core.IntLiteral To: UInt(32)
// CHECK:STDERR: From: Core.IntLiteral To: Int(32)
// CHECK:STDERR: From: Core.IntLiteral To: UInt(32)
// CHECK:STDERR: From: type To: {.a: i32}From: From:{{ }}
// CHECK:STDERR: Core.IntLiteralCore.IntLiteralFrom:  To:  To: Core.IntLiteralUInt(32) To: UInt(32)
// CHECK:STDERR:
// CHECK:STDERR: UInt(32)
// CHECK:STDERR: From: From: Core.IntLiteralCore.IntLiteral To:  To: UInt(32)Int(32)
// CHECK:STDERR: From:{{ }}
// CHECK:STDERR: Core.IntLiteral To: From: Int(32)Core.IntLiteral
// CHECK:STDERR:  To: Int(32)
// CHECK:STDERR: From: Core.IntLiteral To: UInt(32)
// CHECK:STDERR: From: Core.IntLiteral To: UInt(32)
// CHECK:STDERR: From: Core.IntLiteral To: UInt(32)
// CHECK:STDERR: From: Core.IntLiteral To: Int(32)
// CHECK:STDERR: error: no input files
// CHECK:STDERR: From: From: Core.IntLiteralCore.IntLiteral To:  To: UInt(32)Int(32)
// CHECK:STDERR:
