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

class ArgToI(T: I) {}

interface IMisuse {
  let Arg: type;
  // CHECK:STDERR: fail_invalid_interface.carbon:[[@LINE+3]]:18-28: error: cannot convert type `Arg` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_invalid_interface.carbon:[[@LINE-5]]:14-17: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  fn Op(self) -> ArgToI(Arg);
}

// --- fail_misusing_invalid_interface.carbon
library "[[@TEST_NAME]]";
// This import boundary is significant.
import library "invalid_interface";

class C {
  // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+2]]:32: error: name `T` not found [NameNotFound]
  // CHECK:STDERR:
  impl as IMisuse where .Arg = T {
    // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+2]]:34: error: name `T` not found [NameNotFound]
    // CHECK:STDERR:
    fn Op(unused self) -> ArgToI(T) {
      return {};
    }
  }
}

// CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+4]]:28-33: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: imported from: fail_misusing_invalid_interface.carbon:[[@LINE-15]]:1 [InImport]
// CHECK:STDERR: fail_invalid_interface.carbon:12:3-29: note: calling function declared here [InCallToEntity]
// CHECK:STDERR:
fn F(generic T: IMisuse) { T.Op(); }

fn G() {
  var c: C;
  // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+4]]:3-6: error: cannot implicitly convert non-type value of type `C` into type implementing `IMisuse` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE+3]]:3-6: note: type `C` does not implement interface `Core.ImplicitAs(IMisuse)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: fail_misusing_invalid_interface.carbon:[[@LINE-6]]:14-23: note: initializing generic parameter `T` declared here [InitializingGenericParam]
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
    // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE+4]]:27-38: error: cannot convert type `ArgC` into type implementing `I` [ConversionFailureTypeToFacet]
    // CHECK:STDERR: imported from: fail_using_invalid_interface.carbon:[[@LINE-7]]:1 [InImport]
    // CHECK:STDERR: fail_invalid_interface.carbon:5:14-17: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR:
    fn Op(unused self) -> ArgToI(ArgC) {
      return {};
    }
  }
}

// CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE+4]]:28-33: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: imported from: fail_using_invalid_interface.carbon:[[@LINE-17]]:1 [InImport]
// CHECK:STDERR: fail_invalid_interface.carbon:12:3-29: note: calling function declared here [InCallToEntity]
// CHECK:STDERR:
fn F(generic T: IMisuse) { T.Op(); }

fn G() {
  var c: C;
  // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE+4]]:3-6: error: cannot implicitly convert non-type value of type `C` into type implementing `IMisuse` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE+3]]:3-6: note: type `C` does not implement interface `Core.ImplicitAs(IMisuse)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: fail_using_invalid_interface.carbon:[[@LINE-6]]:14-23: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(c);
}
