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
  // CHECK:STDERR:        ╭─┤ fail_overload_no_matching_arity.carbon:[[@LINE+24]]:3
  // CHECK:STDERR: ->     │   override fn f(self);
  // CHECK:STDERR:        ·   ━━━━━━━━━━┳━━━━━━━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: override without compatible virtual in base class
  // CHECK:STDERR:        ·                ┃        [OverrideWithoutVirtualInBase]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ ./overload.h
  // CHECK:STDERR:        │   virtual void f(X x);
  // CHECK:STDERR:        ·                ┬
  // CHECK:STDERR:        ·                ╰──┤ base class function has more parameters (1 vs 0 excluding
  // CHECK:STDERR:        ·                   │ `self`)
  // CHECK:STDERR:        ·                   │ [OverrideCandidateArityMismatch]
  // CHECK:STDERR:        │   virtual void f(Y y);
  // CHECK:STDERR:        ·                ┬
  // CHECK:STDERR:        ·                ╰──┤ base class function has more parameters (1 vs 0 excluding
  // CHECK:STDERR:        ·                   │ `self`)
  // CHECK:STDERR:        ·                   │ [OverrideCandidateArityMismatch]
  // CHECK:STDERR:        │   virtual void f(X x, Y y);
  // CHECK:STDERR:        ·                ┬
  // CHECK:STDERR:        ·                ╰──┤ base class function has more parameters (2 vs 0 excluding
  // CHECK:STDERR:        ·                   │ `self`)
  // CHECK:STDERR:        ·                   │ [OverrideCandidateArityMismatch]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  override fn f(self);
}

class Overload3Params {
  extend base: Cpp.OverloadedBase;
  //
  // CHECK:STDERR:        ╭─┤ fail_overload_no_matching_arity.carbon:[[@LINE+24]]:3
  // CHECK:STDERR: ->     │   override fn f(self, x: Cpp.X, y: Cpp.Y, z: Cpp.X);
  // CHECK:STDERR:        ·   ━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                    ┗━━┫ error: override without compatible virtual in base class
  // CHECK:STDERR:        ·                       ┃        [OverrideWithoutVirtualInBase]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ ./overload.h
  // CHECK:STDERR:        │   virtual void f(X x);
  // CHECK:STDERR:        ·                ┬
  // CHECK:STDERR:        ·                ╰──┤ base class function has fewer parameters (1 vs 3 excluding
  // CHECK:STDERR:        ·                   │ `self`)
  // CHECK:STDERR:        ·                   │ [OverrideCandidateArityMismatch]
  // CHECK:STDERR:        │   virtual void f(Y y);
  // CHECK:STDERR:        ·                ┬
  // CHECK:STDERR:        ·                ╰──┤ base class function has fewer parameters (1 vs 3 excluding
  // CHECK:STDERR:        ·                   │ `self`)
  // CHECK:STDERR:        ·                   │ [OverrideCandidateArityMismatch]
  // CHECK:STDERR:        │   virtual void f(X x, Y y);
  // CHECK:STDERR:        ·                ┬
  // CHECK:STDERR:        ·                ╰──┤ base class function has fewer parameters (2 vs 3 excluding
  // CHECK:STDERR:        ·                   │ `self`)
  // CHECK:STDERR:        ·                   │ [OverrideCandidateArityMismatch]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  override fn f(self, x: Cpp.X, y: Cpp.Y, z: Cpp.X);
}

// --- fail_overload_type.carbon
// TODO: The conversion diagnostics here should have a location.

library "[[@TEST_NAME]]";

import Cpp library "overload.h";

class OverloadX {
  extend base: Cpp.OverloadedBase;
  //
  // CHECK:STDERR:        ╭─┤ fail_overload_type.carbon:[[@LINE+32]]:3
  // CHECK:STDERR: ->     │   override fn f(self, x: Cpp.X);
  // CHECK:STDERR:        ·   ━━━━━━━━━━━━━━━┳━━━━┯━━━━━━━━━
  // CHECK:STDERR:        ·                  ┗━━━━━━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                  │    │  ┃        `Cpp.Y` to `Cpp.X`
  // CHECK:STDERR:        ·                  │    │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                  ╰─────────┤ type `Cpp.Y` does not implement interface
  // CHECK:STDERR:        ·                       │    │ `Core.ImplicitAs(Cpp.X)`
  // CHECK:STDERR:        ·                       │    │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ·                       ╰──────┤ initializing function parameter
  // CHECK:STDERR:        ·                              │ [InCallToFunctionParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_overload_type.carbon:[[@LINE+19]]:3
  // CHECK:STDERR: ->     │   override fn f(self, x: Cpp.X);
  // CHECK:STDERR:        ·   ━━━━━━━━━━━━━━━┳━━━━┯━━━━━━━━━
  // CHECK:STDERR:        ·                  ┗━━━━━━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                  │    │  ┃        `Cpp.Y` to `Cpp.X`
  // CHECK:STDERR:        ·                  │    │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                  ╰─────────┤ type `Cpp.Y` does not implement interface
  // CHECK:STDERR:        ·                       │    │ `Core.ImplicitAs(Cpp.X)`
  // CHECK:STDERR:        ·                       │    │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ·                       ╰──────┤ initializing function parameter
  // CHECK:STDERR:        ·                              │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ ./overload.h
  // CHECK:STDERR:        │   virtual void f(Y y);
  // CHECK:STDERR:        ·                ┬
  // CHECK:STDERR:        ·                ╰──┤ while building thunk to match the signature of this function
  // CHECK:STDERR:        ·                   │ [ThunkSignature]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  override fn f(self, x: Cpp.X);
}

class OverloadY {
  extend base: Cpp.OverloadedBase;
  //
  // CHECK:STDERR:        ╭─┤ fail_overload_type.carbon:[[@LINE+32]]:3
  // CHECK:STDERR: ->     │   override fn f(self, y: Cpp.Y);
  // CHECK:STDERR:        ·   ━━━━━━━━━━━━━━━┳━━━━┯━━━━━━━━━
  // CHECK:STDERR:        ·                  ┗━━━━━━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                  │    │  ┃        `Cpp.X` to `Cpp.Y`
  // CHECK:STDERR:        ·                  │    │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                  ╰─────────┤ type `Cpp.X` does not implement interface
  // CHECK:STDERR:        ·                       │    │ `Core.ImplicitAs(Cpp.Y)`
  // CHECK:STDERR:        ·                       │    │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ·                       ╰──────┤ initializing function parameter
  // CHECK:STDERR:        ·                              │ [InCallToFunctionParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_overload_type.carbon:[[@LINE+19]]:3
  // CHECK:STDERR: ->     │   override fn f(self, y: Cpp.Y);
  // CHECK:STDERR:        ·   ━━━━━━━━━━━━━━━┳━━━━┯━━━━━━━━━
  // CHECK:STDERR:        ·                  ┗━━━━━━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                  │    │  ┃        `Cpp.X` to `Cpp.Y`
  // CHECK:STDERR:        ·                  │    │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                  ╰─────────┤ type `Cpp.X` does not implement interface
  // CHECK:STDERR:        ·                       │    │ `Core.ImplicitAs(Cpp.Y)`
  // CHECK:STDERR:        ·                       │    │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ·                       ╰──────┤ initializing function parameter
  // CHECK:STDERR:        ·                              │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ ./overload.h
  // CHECK:STDERR:        │   virtual void f(X x);
  // CHECK:STDERR:        ·                ┬
  // CHECK:STDERR:        ·                ╰──┤ while building thunk to match the signature of this function
  // CHECK:STDERR:        ·                   │ [ThunkSignature]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  override fn f(self, y: Cpp.Y);
}
