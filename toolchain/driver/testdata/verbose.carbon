// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// ARGS: -v compile --phase=codegen --target=x86_64-unknown-linux-gnu --output=%t %s
//
// Verifies that various phases are included in vlog output.
//
// SET-CHECK-SUBSET
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/verbose.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/verbose.carbon
// CHECK:STDERR: *** SourceBuffer ***
// CHECK:STDERR: *** Lex::Lex: empty.carbon ***
// CHECK:STDERR: *** Lex::TokenizedBuffer ***
// CHECK:STDERR: - filename: empty.carbon
// CHECK:STDERR: *** Lex::Lex done ***
// CHECK:STDERR: *** Parse::Parse: empty.carbon ***
// CHECK:STDERR: *** Parse::Tree ***
// CHECK:STDERR: - filename: empty.carbon
// CHECK:STDERR: *** Parse::Parse done ***
// CHECK:STDERR: *** Check::CheckParseTrees ***
// CHECK:STDERR: *** SemIR::File ***
// CHECK:STDERR: --- empty.carbon
// CHECK:STDERR: *** Check::CheckParseTrees done ***
// CHECK:STDERR: *** Lower::LowerToLLVM: empty.carbon ***
// CHECK:STDERR: *** llvm::Module ***
// CHECK:STDERR: ; ModuleID = 'empty.carbon'
// CHECK:STDERR: source_filename = "empty.carbon"
// CHECK:STDERR: *** Lower::LowerToLLVM done ***
// CHECK:STDERR: *** CodeGen: empty.carbon ***
// CHECK:STDERR: *** Assembly ***
// CHECK:STDERR: {{\t}}.file{{\t}}"empty.carbon"
// CHECK:STDERR: *** CodeGen done ***

// --- empty.carbon
