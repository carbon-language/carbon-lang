// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/stringify.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/stringify.carbon

// --- fail_empty_params.carbon

library "[[@TEST_NAME]]";

class NoParams {}
class EmptyParams() {}

var v: NoParams;
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `NoParams` to `EmptyParams()` [ConversionFailure]
// CHECK:STDERR: var w: EmptyParams() = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_empty_params.carbon:[[@LINE+4]]:1: note: type `NoParams` does not implement interface `Core.ImplicitAs(EmptyParams())` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var w: EmptyParams() = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var w: EmptyParams() = v;

// --- fail_nested.carbon

library "[[@TEST_NAME]]";

class Outer(T:! type) {
  class Inner(U:! type) {
  }
}

var v: Outer({}*);

// TODO: It would be nice to include the `Outer({}*).` prefix in the name of `Inner`.
// CHECK:STDERR: fail_nested.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `Outer({}*)` to `Inner({.a: i32}*)` [ConversionFailure]
// CHECK:STDERR: var w: Outer({}*).Inner({.a: i32}*) = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:1: note: type `Outer({}*)` does not implement interface `Core.ImplicitAs(Inner({.a: i32}*))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var w: Outer({}*).Inner({.a: i32}*) = v;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var w: Outer({}*).Inner({.a: i32}*) = v;

// --- fail_int_value.carbon

library "[[@TEST_NAME]]";

class C(N:! i32) {}

// CHECK:STDERR: fail_int_value.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `()` to `C(123)` [ConversionFailure]
// CHECK:STDERR: var v: C(123) = ();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_int_value.carbon:[[@LINE+4]]:1: note: type `()` does not implement interface `Core.ImplicitAs(C(123))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var v: C(123) = ();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
var v: C(123) = ();

// --- fail_class_param.carbon

library "[[@TEST_NAME]]";

class D {
  var a: i32;
  var b: i32;
}

class E(F:! D) {}

// CHECK:STDERR: fail_class_param.carbon:[[@LINE+14]]:8: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
// CHECK:STDERR: var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_class_param.carbon:[[@LINE-5]]:9: note: initializing generic parameter `F` declared here [InitializingGenericParam]
// CHECK:STDERR: class E(F:! D) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_class_param.carbon:[[@LINE+7]]:36: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
// CHECK:STDERR: var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
// CHECK:STDERR:                                    ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_class_param.carbon:[[@LINE-12]]:9: note: initializing generic parameter `F` declared here [InitializingGenericParam]
// CHECK:STDERR: class E(F:! D) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
var g: E({.a = 1, .b = 2}) = {} as E({.a = 3, .b = 4} as D);
