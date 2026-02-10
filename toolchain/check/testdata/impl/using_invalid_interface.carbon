// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/using_invalid_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/using_invalid_interface.carbon

// --- fail_invalid_interface.carbon
library "[[@TEST_NAME]]";

interface I {}

class ArgToI(T:! I) {}

interface IMisuse {
  let Arg:! type;
  // CHECK:STDERR: fail_invalid_interface.carbon:[[@LINE+7]]:26: error: cannot convert type `Arg` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   fn Op[self: Self]() -> ArgToI(Arg);
  // CHECK:STDERR:                          ^~~~~~~~~~~
  // CHECK:STDERR: fail_invalid_interface.carbon:[[@LINE-7]]:14: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: class ArgToI(T:! I) {}
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  fn Op[self: Self]() -> ArgToI(Arg);
}

// --- fail_misusing_invalid_interface.carbon
library "[[@TEST_NAME]]";
// This import boundary is significant.
import library "invalid_interface";

class C {
  // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+4]]:32: error: name `T` not found [NameNotFound]
  // CHECK:STDERR:   impl as IMisuse where .Arg = T {
  // CHECK:STDERR:                                ^
  // CHECK:STDERR:
  impl as IMisuse where .Arg = T {
    // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+4]]:35: error: name `T` not found [NameNotFound]
    // CHECK:STDERR:     fn Op[self: Self]() -> ArgToI(T) {
    // CHECK:STDERR:                                   ^
    // CHECK:STDERR:
    fn Op[self: Self]() -> ArgToI(T) {
      return {};
    }
  }
}

// CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+8]]:21: error: missing object argument in method call [MissingObjectInMethodCall]
// CHECK:STDERR: fn F(T:! IMisuse) { T.Op(); }
// CHECK:STDERR:                     ^~~~~~
// CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE-21]]:1: in import [InImport]
// CHECK:STDERR: fail_invalid_interface.carbon:16:3: note: calling function declared here [InCallToFunction]
// CHECK:STDERR:   fn Op[self: Self]() -> ArgToI(Arg);
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! IMisuse) { T.Op(); }

fn G() {
  var c: C;
  // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+10]]:3: error: cannot implicitly convert non-type value of type `C` into type implementing `IMisuse` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   F(c);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+7]]:3: note: type `C` does not implement interface `Core.ImplicitAs(IMisuse)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   F(c);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE-10]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! IMisuse) { T.Op(); }
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(c);
}

// --- fail_using_invalid_interface.carbon
library "[[@TEST_NAME]]";
// This import boundary is significant.
import library "invalid_interface";

class ArgC {}

class C {
  impl as IMisuse where .Arg = ArgC {
    // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE+8]]:28: error: cannot convert type `ArgC` into type implementing `I` [ConversionFailureTypeToFacet]
    // CHECK:STDERR:     fn Op[self: Self]() -> ArgToI(ArgC) {
    // CHECK:STDERR:                            ^~~~~~~~~~~~
    // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE-9]]:1: in import [InImport]
    // CHECK:STDERR: fail_invalid_interface.carbon:5:14: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR: class ArgToI(T:! I) {}
    // CHECK:STDERR:              ^
    // CHECK:STDERR:
    fn Op[self: Self]() -> ArgToI(ArgC) {
      return {};
    }
  }
}

// CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE+8]]:21: error: missing object argument in method call [MissingObjectInMethodCall]
// CHECK:STDERR: fn F(T:! IMisuse) { T.Op(); }
// CHECK:STDERR:                     ^~~~~~
// CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE-23]]:1: in import [InImport]
// CHECK:STDERR: fail_invalid_interface.carbon:16:3: note: calling function declared here [InCallToFunction]
// CHECK:STDERR:   fn Op[self: Self]() -> ArgToI(Arg);
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! IMisuse) { T.Op(); }

fn G() {
  var c: C;
  // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE+10]]:3: error: cannot implicitly convert non-type value of type `C` into type implementing `IMisuse` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   F(c);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE+7]]:3: note: type `C` does not implement interface `Core.ImplicitAs(IMisuse)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   F(c);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE-10]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(T:! IMisuse) { T.Op(); }
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  F(c);
}
