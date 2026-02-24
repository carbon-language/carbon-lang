// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/non_type_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/non_type_param.carbon

// --- templates.h

struct A {};

template<int N> struct Simple {};
Simple<123> MakeSimple123() { return Simple<123>(); }
constexpr int Constant123 = 123;

template<int M, int N> struct TwoNonType {};
template<typename T, T N> struct DependentNonType {};

// --- valid_int_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

var s: Cpp.Simple(123) = Cpp.MakeSimple123();

// --- fail_int_param_mismatch.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_int_param_mismatch.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `Cpp.Simple` to `Cpp.Simple` [ConversionFailure]
// CHECK:STDERR: var s: Cpp.Simple(456) = Cpp.MakeSimple123();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_int_param_mismatch.carbon:[[@LINE+4]]:1: note: type `Cpp.Simple` does not implement interface `Core.ImplicitAs(Cpp.Simple)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var s: Cpp.Simple(456) = Cpp.MakeSimple123();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var s: Cpp.Simple(456) = Cpp.MakeSimple123();

// --- nonliteral.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

var s: Cpp.Simple(100 + 20 + 3) = Cpp.MakeSimple123();

// --- constant.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

var s: Cpp.Simple(Cpp.Constant123) = Cpp.MakeSimple123();

// --- conversion.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

class C { adapt {}; }
impl C as Core.ImplicitAs(i32) {
  eval fn Convert[unused self: Self]() -> i32 { return 123; }
}
var s: Cpp.Simple({} as C) = Cpp.MakeSimple123();

// --- valid.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

//@dump-sem-ir-begin
var x: Cpp.TwoNonType(1, 2);
//@dump-sem-ir-end

// --- fail_todo_dependent.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

var n: i32;

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_dependent.carbon:[[@LINE+4]]:34: error: semantics TODO: `unsupported argument type for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: var x: Cpp.DependentNonType(i32, 1);
// CHECK:STDERR:                                  ^
// CHECK:STDERR:
var x: Cpp.DependentNonType(i32, 1);

// CHECK:STDERR: fail_todo_dependent.carbon:[[@LINE+4]]:35: error: semantics TODO: `unsupported argument type for non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: var y: Cpp.DependentNonType(i32*, &n);
// CHECK:STDERR:                                   ^~
// CHECK:STDERR:
var y: Cpp.DependentNonType(i32*, &n);
//@dump-sem-ir-end

// --- fail_type_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

// CHECK:STDERR: fail_type_argument.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `type` to `i32` [ConversionFailure]
// CHECK:STDERR: var x: Cpp.TwoNonType(1, Cpp.A);
// CHECK:STDERR:                          ^~~~~
// CHECK:STDERR: fail_type_argument.carbon:[[@LINE+4]]:26: note: type `type` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: Cpp.TwoNonType(1, Cpp.A);
// CHECK:STDERR:                          ^~~~~
// CHECK:STDERR:
var x: Cpp.TwoNonType(1, Cpp.A);

// --- fail_no_conversion.carbon

library "[[@TEST_NAME]]";

import Cpp library "templates.h";

var n: i32;

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `i32*` to `i32` [ConversionFailure]
// CHECK:STDERR: var x: Cpp.TwoNonType(1, &n);
// CHECK:STDERR:                          ^~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+4]]:26: note: type `i32*` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: Cpp.TwoNonType(1, &n);
// CHECK:STDERR:                          ^~
// CHECK:STDERR:
var x: Cpp.TwoNonType(1, &n);

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TwoNonType: type = class_type @TwoNonType [concrete]
// CHECK:STDOUT:   %pattern_type.8a6: type = pattern_type %TwoNonType [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.8a6 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.8a6 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %TwoNonType = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref %TwoNonType = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_dependent.carbon
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
