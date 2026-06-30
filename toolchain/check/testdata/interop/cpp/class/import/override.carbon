// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/override.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/override.carbon

// --- overload.h

struct X {};
struct Y {};

struct OverloadedBase {
  virtual void f(X x);
  virtual void f(Y y);
  virtual void f(X x, Y y);
};

// --- overload_arity.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload.h";

class Overload2Params {
  extend base: Cpp.OverloadedBase;
  override fn f(self, x: Cpp.X, y: Cpp.Y);
}

// --- fail_overload_no_matching_arity.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload.h";

class Overload0Params {
  extend base: Cpp.OverloadedBase;
  //
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE+16]]:3: error: override without compatible virtual in base class [OverrideWithoutVirtualInBase]
  // CHECK:STDERR:   override fn f(self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:6:16: note: base class function has more parameters (1 vs 0 excluding `self`) [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(X x);
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-12]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:7:16: note: base class function has more parameters (1 vs 0 excluding `self`) [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(Y y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:8:16: note: base class function has more parameters (2 vs 0 excluding `self`) [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(X x, Y y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  override fn f(self);
}

class Overload3Params {
  extend base: Cpp.OverloadedBase;
  //
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE+16]]:3: error: override without compatible virtual in base class [OverrideWithoutVirtualInBase]
  // CHECK:STDERR:   override fn f(self, x: Cpp.X, y: Cpp.Y, z: Cpp.X);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-30]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:6:16: note: base class function has fewer parameters (1 vs 3 excluding `self`) [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(X x);
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-34]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:7:16: note: base class function has fewer parameters (1 vs 3 excluding `self`) [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(Y y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-38]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:8:16: note: base class function has fewer parameters (2 vs 3 excluding `self`) [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(X x, Y y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  override fn f(self, x: Cpp.X, y: Cpp.Y, z: Cpp.X);
}

// --- fail_overload_type.carbon
// CHECK:STDERR: fail_overload_type.carbon: error: cannot implicitly convert expression of type `Cpp.Y` to `Cpp.X` [ConversionFailure]
// CHECK:STDERR: fail_overload_type.carbon: note: type `Cpp.Y` does not implement interface `Core.ImplicitAs(Cpp.X)` [MissingImplInMemberAccessInContext]
// TODO: The conversion diagnostics here should have a location.

library "[[@TEST_NAME]]";

import Cpp library "overload.h";

class OverloadX {
  extend base: Cpp.OverloadedBase;
  //
  // CHECK:STDERR: fail_overload_type.carbon:[[@LINE+10]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   override fn f(self, x: Cpp.X);
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR: fail_overload_type.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:7:16: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   virtual void f(Y y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_overload_type.carbon: error: cannot implicitly convert expression of type `Cpp.X` to `Cpp.Y` [ConversionFailure]
  // CHECK:STDERR: fail_overload_type.carbon: note: type `Cpp.X` does not implement interface `Core.ImplicitAs(Cpp.Y)` [MissingImplInMemberAccessInContext]
  override fn f(self, x: Cpp.X);
}

class OverloadY {
  extend base: Cpp.OverloadedBase;
  //
  // CHECK:STDERR: fail_overload_type.carbon:[[@LINE+8]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   override fn f(self, y: Cpp.Y);
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR: fail_overload_type.carbon:[[@LINE-24]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:6:16: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   virtual void f(X x);
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  override fn f(self, y: Cpp.Y);
}
