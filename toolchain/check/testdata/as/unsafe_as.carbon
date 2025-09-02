// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/unsafe_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/unsafe_as.carbon

// --- qualifiers.carbon

library "[[@TEST_NAME]]";

class X {}

fn Convert(p: const X*) -> X* {
  //@dump-sem-ir-begin
  return p unsafe as X*;
  //@dump-sem-ir-end
}

// --- fail_no_conversion.carbon

library "[[@TEST_NAME]]";

class A {};
class B {};

fn Convert(a: A) -> B {
  // CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+7]]:10: error: cannot convert expression of type `A` to `B` with `as` [ConversionFailure]
  // CHECK:STDERR:   return a unsafe as B;
  // CHECK:STDERR:          ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+4]]:10: note: type `A` does not implement interface `Core.UnsafeAs(B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a unsafe as B;
  // CHECK:STDERR:          ^~~~~~~~~~~~~
  // CHECK:STDERR:
  return a unsafe as B;
}

// --- fail_remove_qualifiers_without_unsafe.carbon

library "[[@TEST_NAME]]";

class X {}

fn Convert(p: const X*) -> X* {
  // CHECK:STDERR: fail_remove_qualifiers_without_unsafe.carbon:[[@LINE+7]]:10: error: cannot convert expression of type `const X*` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   return p as X*;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_remove_qualifiers_without_unsafe.carbon:[[@LINE+4]]:10: note: type `const X*` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p as X*;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return p as X*;
}

// CHECK:STDOUT: --- qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %const: type = const_type %X [concrete]
// CHECK:STDOUT:   %ptr.cbd: type = ptr_type %const [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert(%p.param: %ptr.cbd) -> %ptr.d17 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.cbd = name_ref p, %p
// CHECK:STDOUT:   %X.ref.loc8: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc8: type = ptr_type %X.ref.loc8 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   %.loc8_19.1: %ptr.d17 = as_compatible %p.ref
// CHECK:STDOUT:   %.loc8_19.2: %ptr.d17 = converted %p.ref, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
