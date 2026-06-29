// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --clang-arg=-Wno-abstract-final-class
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/abstract.carbon

// --- abstract.h

struct A {
  virtual void f() = 0;
};

// --- derive_base_from_abstract.carbon

library "[[@TEST_NAME]]";

import Cpp library "abstract.h";

base class B {
  extend base: Cpp.A;
}

// --- todo_fail_derive_final_from_abstract.carbon

library "[[@TEST_NAME]]";

import Cpp library "abstract.h";

class C {
  // TODO: We should reject this because `f` is not implemented.
  extend base: Cpp.A;
}

// --- impl_abstract_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "abstract.h";

class C {
  extend base: Cpp.A;
  override fn f(unused ref self) {}
}

// --- fail_impl_mismatch_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "abstract.h";

class C {
  extend base: Cpp.A;
  // CHECK:STDERR: fail_impl_mismatch_member.carbon:[[@LINE+4]]:3: error: override without compatible virtual in base class [OverrideWithoutVirtualInBase]
  // CHECK:STDERR:   override fn invalid(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  override fn invalid(unused ref self) {}
}

// --- abstract_final.h

// C++ allows a class to be both abstract and final.
struct AbstractFinal final {
  virtual void f() = 0;
};

// --- fail_derive_from_abstract_final.carbon

library "[[@TEST_NAME]]";

import Cpp library "abstract_final.h";

class C {
  // CHECK:STDERR: fail_derive_from_abstract_final.carbon:[[@LINE+4]]:16: error: deriving from final type `AbstractFinal`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: Cpp.AbstractFinal;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Cpp.AbstractFinal;
}

// --- use_abstract_final.carbon

library "[[@TEST_NAME]]";

import Cpp library "abstract_final.h";

// We model an abstract final class as being final, rather than abstract. This
// prevents inheritance from it, and there's no actual way to initialize an
// instance of the type, but that does not prevent defining a function that
// returns the type by value. Make sure that doesn't crash.
//@dump-sem-ir-begin
fn F() -> Cpp.AbstractFinal {
  return F();
}
//@dump-sem-ir-end


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
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE+16]]:3: error: override without compatible virtual in base class [OverrideWithoutVirtualInBase]
  // CHECK:STDERR:   override fn f(self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:6:16: note: candidate function has 2 parameters including `self`, but override has only 1 [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(X x);
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:7:16: note: candidate function has 2 parameters including `self`, but override has only 1 [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(Y y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-15]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:8:16: note: candidate function has 3 parameters including `self`, but override has only 1 [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(X x, Y y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  override fn f(self);
}

class Overload3Params {
  extend base: Cpp.OverloadedBase;
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE+16]]:3: error: override without compatible virtual in base class [OverrideWithoutVirtualInBase]
  // CHECK:STDERR:   override fn f(self, x: Cpp.X, y: Cpp.Y, z: Cpp.X);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-28]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:6:16: note: candidate function has 2 parameters including `self`, but override has 4 [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(X x);
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-32]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:7:16: note: candidate function has 2 parameters including `self`, but override has 4 [OverrideCandidateArityMismatch]
  // CHECK:STDERR:   virtual void f(Y y);
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_overload_no_matching_arity.carbon:[[@LINE-36]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:8:16: note: candidate function has 3 parameters including `self`, but override has 4 [OverrideCandidateArityMismatch]
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
  // CHECK:STDERR: fail_overload_type.carbon:[[@LINE+10]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   override fn f(self, x: Cpp.X);
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR: fail_overload_type.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
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
  // CHECK:STDERR: fail_overload_type.carbon:[[@LINE+8]]:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   override fn f(self, y: Cpp.Y);
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR: fail_overload_type.carbon:[[@LINE-22]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload.h:6:16: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   virtual void f(X x);
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  override fn f(self, y: Cpp.Y);
}

// CHECK:STDOUT: --- use_abstract_final.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AbstractFinal: type = class_type @AbstractFinal [concrete]
// CHECK:STDOUT:   %.dbd: Core.Form = init_form %AbstractFinal [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %AbstractFinal [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type = return_slot_pattern %return.param_patt, %AbstractFinal [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %AbstractFinal.f.type: type = fn_type @AbstractFinal.f [concrete]
// CHECK:STDOUT:   %AbstractFinal.f: %AbstractFinal.f.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .AbstractFinal = %AbstractFinal.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AbstractFinal.decl: type = class_decl @AbstractFinal [concrete = constants.%AbstractFinal] {} {}
// CHECK:STDOUT:   %AbstractFinal.f.decl: %AbstractFinal.f.type = fn_decl @AbstractFinal.f [concrete = constants.%AbstractFinal.f] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern %return.param_patt, %AbstractFinal.ref [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %AbstractFinal.ref: type = name_ref AbstractFinal, imports.%AbstractFinal.decl [concrete = constants.%AbstractFinal]
// CHECK:STDOUT:     %.loc11_14.2: Core.Form = init_form %AbstractFinal.ref [concrete = constants.%.dbd]
// CHECK:STDOUT:     %return.param: ref %AbstractFinal = out_param call_param0
// CHECK:STDOUT:     %return: ref %AbstractFinal = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @AbstractFinal.vtable {
// CHECK:STDOUT:   imports.%AbstractFinal.f.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %AbstractFinal {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc11_14.1: ref %AbstractFinal = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %AbstractFinal to %.loc11_14.1 = call %F.ref()
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
