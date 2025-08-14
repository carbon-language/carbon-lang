// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --clang-arg=-Wno-abstract-final-class
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/abstract.carbon

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

// --- fail_todo_impl_abstract_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "abstract.h";

class C {
  extend base: Cpp.A;
  // CHECK:STDERR: fail_todo_impl_abstract_member.carbon:[[@LINE+4]]:3: error: impl without compatible virtual in base class [ImplWithoutVirtualInBase]
  // CHECK:STDERR:   impl fn f[addr self: Self*]() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl fn f[addr self: Self*]() {}
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

// CHECK:STDOUT: --- use_abstract_final.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AbstractFinal: type = class_type @AbstractFinal [concrete]
// CHECK:STDOUT:   %pattern_type.2e0: type = pattern_type %AbstractFinal [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.4da: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%AbstractFinal) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.0fa: %T.as.Destroy.impl.Op.type.4da = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d4c: type = ptr_type %AbstractFinal [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .AbstractFinal = %AbstractFinal.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AbstractFinal.decl: type = class_decl @AbstractFinal [concrete = constants.%AbstractFinal] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.2e0 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.2e0 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %AbstractFinal.ref: type = name_ref AbstractFinal, imports.%AbstractFinal.decl [concrete = constants.%AbstractFinal]
// CHECK:STDOUT:     %return.param: ref %AbstractFinal = out_param call_param0
// CHECK:STDOUT:     %return: ref %AbstractFinal = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param: %AbstractFinal {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc11: ref %AbstractFinal = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %AbstractFinal = call %F.ref() to %.loc11
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc11, constants.%T.as.Destroy.impl.Op.0fa
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc11, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d4c = addr_of %.loc11
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
