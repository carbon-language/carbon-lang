// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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

  // CHECK:STDERR: fail_todo_copy_enum.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   Cpp.C.C().F(Cpp.C.e);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_copy_enum.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %.c3b: type = overloaded_cpp_fn_type @C.C [concrete]
// CHECK:STDOUT:   %empty_struct.7f4: %.c3b = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.81a: %.4f0 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.793: type = ptr_type %.4f0 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.1: type = fn_type @F__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.1: %F__carbon_thunk.type.eda1ac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.bdd: type = overloaded_cpp_fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %empty_struct.255: %.bdd = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.bb7: type = class_type @.2 [concrete]
// CHECK:STDOUT:   %.f96: type = overloaded_cpp_fn_type @C.F [concrete]
// CHECK:STDOUT:   %empty_struct.9b5: %.f96 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.1d6: %.bb7 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.73d: type = ptr_type %.bb7 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.2: type = fn_type @F__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.2: %F__carbon_thunk.type.eda1ac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.8e9: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%.bb7) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.e52: %T.as.Destroy.impl.Op.type.8e9 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1b3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.21b: %T.as.Destroy.impl.Op.type.1b3 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d1b: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%.4f0) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.002: %T.as.Destroy.impl.Op.type.d1b = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .F = %.815
// CHECK:STDOUT:     .b = %int_1.81a
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.815: %.c3b = overloaded_cpp_fn_decl @C.C [concrete = constants.%empty_struct.7f4]
// CHECK:STDOUT:   %int_1.81a: %.4f0 = int_value 1 [concrete = constants.%int_1.81a]
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.1: %F__carbon_thunk.type.eda1ac.1 = fn_decl @F__carbon_thunk.1 [concrete = constants.%F__carbon_thunk.0cd6a8.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %.481: %.bdd = overloaded_cpp_fn_decl @C__carbon_thunk [concrete = constants.%empty_struct.255]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.146: %.f96 = overloaded_cpp_fn_decl @C.F [concrete = constants.%empty_struct.9b5]
// CHECK:STDOUT:   %int_1.1d6: %.bb7 = int_value 1 [concrete = constants.%int_1.1d6]
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.2: %F__carbon_thunk.type.eda1ac.2 = fn_decl @F__carbon_thunk.2 [concrete = constants.%F__carbon_thunk.0cd6a8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %F.ref.loc8: %.c3b = name_ref F, imports.%.815 [concrete = constants.%empty_struct.7f4]
// CHECK:STDOUT:   %Cpp.ref.loc8_9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %b.ref: %.4f0 = name_ref b, imports.%int_1.81a [concrete = constants.%int_1.81a]
// CHECK:STDOUT:   %.loc8_12.1: ref %.4f0 = temporary_storage
// CHECK:STDOUT:   %.loc8_12.2: ref %.4f0 = temporary %.loc8_12.1, %b.ref
// CHECK:STDOUT:   %addr.loc8_14: %ptr.793 = addr_of %.loc8_12.2
// CHECK:STDOUT:   %F__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.e1b8ec.1(%addr.loc8_14)
// CHECK:STDOUT:   %Cpp.ref.loc15_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc15_6: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc15_8: %.bdd = name_ref C, imports.%.481 [concrete = constants.%empty_struct.255]
// CHECK:STDOUT:   %.loc15_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc15_11.1: %ptr.d9e = addr_of %.loc15_11.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc15_11.1)
// CHECK:STDOUT:   %.loc15_11.2: init %C = in_place_init %C__carbon_thunk.call, %.loc15_11.1
// CHECK:STDOUT:   %.loc15_11.3: ref %C = temporary %.loc15_11.1, %.loc15_11.2
// CHECK:STDOUT:   %F.ref.loc15: %.f96 = name_ref F, imports.%.146 [concrete = constants.%empty_struct.9b5]
// CHECK:STDOUT:   %Cpp.ref.loc15_15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc15_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e.ref: %.bb7 = name_ref e, imports.%int_1.1d6 [concrete = constants.%int_1.1d6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_20.1: ref %.bb7 = temporary_storage
// CHECK:STDOUT:   %.loc15_20.2: ref %.bb7 = temporary %.loc15_20.1, %e.ref
// CHECK:STDOUT:   %addr.loc15_22: %ptr.73d = addr_of %.loc15_20.2
// CHECK:STDOUT:   %F__carbon_thunk.call.loc15: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.e1b8ec.2(<error>, %addr.loc15_22)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc15_20: <bound method> = bound_method %.loc15_20.1, constants.%T.as.Destroy.impl.Op.e52
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc15_20: <bound method> = bound_method %.loc15_20.1, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc15_20: %ptr.73d = addr_of %.loc15_20.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc15_20: init %empty_tuple.type = call %bound_method.loc15_20(%addr.loc15_20)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc15_11: <bound method> = bound_method %.loc15_11.1, constants.%T.as.Destroy.impl.Op.21b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc15_11: <bound method> = bound_method %.loc15_11.1, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc15_11.2: %ptr.d9e = addr_of %.loc15_11.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc15_11: init %empty_tuple.type = call %bound_method.loc15_11(%addr.loc15_11.2)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.1, constants.%T.as.Destroy.impl.Op.002
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.1, %T.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8_12: %ptr.793 = addr_of %.loc8_12.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
