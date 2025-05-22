// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/bad_conversions.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/bad_conversions.carbon

// --- fail_compound_type_mismatch.carbon

library "[[@TEST_NAME]]";

class A {
  var a: {};
}

class B {
  var b: {};
}

fn AccessBInA(a: A) -> {} {
  // CHECK:STDERR: fail_compound_type_mismatch.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `A` to `B` [ConversionFailure]
  // CHECK:STDERR:   return a.(B.b);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_compound_type_mismatch.carbon:[[@LINE+4]]:10: note: type `A` does not implement interface `Core.ImplicitAs(B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.(B.b);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return a.(B.b);
}

// --- fail_convert_to_invalid.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_convert_to_invalid.carbon:[[@LINE+4]]:10: error: name `NoSuchType` not found [NameNotFound]
  // CHECK:STDERR:   var a: NoSuchType;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  var a: NoSuchType;
}

fn Make() -> C {
  return {.a = 123};
}

// --- fail_derived_to_base.carbon

base class A1 {
  var a: ();
}

base class A2 {
  var a: ();
}

class B2 {
  extend base: A2;
  var b: ();
}

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+7]]:38: error: cannot implicitly convert expression of type `B2*` to `A1*` [ConversionFailure]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:38: note: type `B2*` does not implement interface `Core.ImplicitAs(A1*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR:
fn ConvertUnrelated(p: B2*) -> A1* { return p; }

class Incomplete;

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+7]]:47: error: cannot implicitly convert expression of type `Incomplete*` to `A2*` [ConversionFailure]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:47: note: type `Incomplete*` does not implement interface `Core.ImplicitAs(A2*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR:
fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
