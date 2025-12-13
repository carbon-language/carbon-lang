// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/template_template_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/template_template_param.carbon

// --- templates.h

template<typename> struct A {};
template<typename> struct B {};

template<template<typename> typename, template<typename> typename>
struct TwoTemplates {};

// --- fail_todo_valid.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_valid.carbon:[[@LINE+4]]:25: error: semantics TODO: `argument for template template parameter` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.TwoTemplates(Cpp.A, Cpp.B);
// CHECK:STDERR:                         ^~~~~
// CHECK:STDERR:
var x: Cpp.TwoTemplates(Cpp.A, Cpp.B);
//@dump-sem-ir-end

// --- fail_non_type_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_non_type_argument.carbon:[[@LINE+4]]:25: error: semantics TODO: `argument for template template parameter` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.TwoTemplates(Cpp.A, true);
// CHECK:STDERR:                         ^~~~~
// CHECK:STDERR:
var x: Cpp.TwoTemplates(Cpp.A, true);

// CHECK:STDOUT: --- fail_todo_valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: <error> = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: <error> = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref <error> = var %x.var_patt [concrete = <error>]
// CHECK:STDOUT:   %x: ref <error> = ref_binding x, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
