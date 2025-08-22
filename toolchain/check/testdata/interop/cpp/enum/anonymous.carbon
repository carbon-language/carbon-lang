// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --clang-arg=-fms-compatibility
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/anonymous.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/anonymous.carbon

// --- anon.h

enum { a, b, c };

void F(decltype(a));

class C {
 public:
  enum { d, e, f };

  void F(decltype(d));
};

// --- fail_todo_copy_enum.carbon

library "[[@TEST_NAME]]";

import Cpp library "anon.h";

//@dump-sem-ir-begin
fn G() {
  Cpp.F(Cpp.b);

  Cpp.C.C().F(Cpp.C.e);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_todo_copy_enum.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.4f0: type = class_type @.1 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.793: type = ptr_type %.4f0 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type: type = fn_type @F__carbon_thunk [concrete]
// CHECK:STDOUT:   %F__carbon_thunk: %F__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.81a: %.4f0 = int_value 1 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.C.type: type = fn_type @C.C [concrete]
// CHECK:STDOUT:   %C.C: %C.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %.bb7: type = class_type @.2 [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.1d6: %.bb7 = int_value 1 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.21b: %T.as.Destroy.impl.Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d1b: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%.4f0) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.002: %T.as.Destroy.impl.Op.type.d1b = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .b = %int_1.81a
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F__carbon_thunk.decl: %F__carbon_thunk.type = fn_decl @F__carbon_thunk [concrete = constants.%F__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.81a: %.4f0 = int_value 1 [concrete = constants.%int_1.81a]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.decl: %C.C.type = fn_decl @C.C [concrete = constants.%C.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.1d6: %.bb7 = int_value 1 [concrete = constants.%int_1.1d6]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %F.ref.loc8: %F.type = name_ref F, imports.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %Cpp.ref.loc8_9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %b.ref: %.4f0 = name_ref b, imports.%int_1.81a [concrete = constants.%int_1.81a]
// CHECK:STDOUT:   %.loc8_12.1: ref %.4f0 = temporary_storage
// CHECK:STDOUT:   %.loc8_12.2: ref %.4f0 = temporary %.loc8_12.1, %b.ref
// CHECK:STDOUT:   %addr.loc8_14: %ptr.793 = addr_of %.loc8_12.2
// CHECK:STDOUT:   %F__carbon_thunk.call: init %empty_tuple.type = call imports.%F__carbon_thunk.decl(%addr.loc8_14)
// CHECK:STDOUT:   %Cpp.ref.loc10_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_6: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_8: %C.C.type = name_ref C, imports.%C.C.decl [concrete = constants.%C.C]
// CHECK:STDOUT:   %.loc10_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %C.C.call: init %C = call %C.ref.loc10_8() to %.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.2: ref %C = temporary %.loc10_11.1, %C.C.call
// CHECK:STDOUT:   %F.ref.loc10: %C.F.type = name_ref F, imports.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.bound: <bound method> = bound_method %.loc10_11.2, %F.ref.loc10
// CHECK:STDOUT:   %Cpp.ref.loc10_15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e.ref: %.bb7 = name_ref e, imports.%int_1.1d6 [concrete = constants.%int_1.1d6]
// CHECK:STDOUT:   %addr.loc10_11.1: %ptr.d9e = addr_of %.loc10_11.2
// CHECK:STDOUT:   %C.F.call: init %empty_tuple.type = call %C.F.bound(%addr.loc10_11.1, %e.ref)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_11.1, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_11.1, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_11.2: %ptr.d9e = addr_of %.loc10_11.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10_11.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.1, constants.%T.as.Destroy.impl.Op.002
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.1, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_12: %ptr.793 = addr_of %.loc8_12.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
