// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/non_type_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/non_type_param.carbon

// --- templates.h

struct A {};

template<int M, int N> struct TwoNonType {};
template<typename T, T N> struct DependentNonType {};

// --- fail_todo_valid.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_valid.carbon:[[@LINE+4]]:23: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.TwoNonType(1, 2);
// CHECK:STDERR:                       ^
// CHECK:STDERR:
var x: Cpp.TwoNonType(1, 2);
//@dump-sem-ir-end

// --- fail_todo_dependent.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

var n: i32;

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_dependent.carbon:[[@LINE+4]]:34: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.DependentNonType(i32, 1);
// CHECK:STDERR:                                  ^
// CHECK:STDERR:
var x: Cpp.DependentNonType(i32, 1);

// CHECK:STDERR: fail_todo_dependent.carbon:[[@LINE+4]]:35: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: var y: Cpp.DependentNonType(i32*, &n);
// CHECK:STDERR:                                   ^~
// CHECK:STDERR:
var y: Cpp.DependentNonType(i32*, &n);
//@dump-sem-ir-end

// --- fail_type_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_type_argument.carbon:[[@LINE+4]]:23: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.TwoNonType(1, Cpp.A);
// CHECK:STDERR:                       ^
// CHECK:STDERR:
var x: Cpp.TwoNonType(1, Cpp.A);

// --- fail_no_conversion.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

var n: i32;

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+4]]:23: error: semantics TODO: `argument for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.TwoNonType(1, &n);
// CHECK:STDERR:                       ^
// CHECK:STDERR:
var x: Cpp.TwoNonType(1, &n);

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
// CHECK:STDOUT: --- fail_todo_dependent.carbon
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
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: <error> = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: <error> = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref <error> = var %y.var_patt [concrete = <error>]
// CHECK:STDOUT:   %y: ref <error> = ref_binding y, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
