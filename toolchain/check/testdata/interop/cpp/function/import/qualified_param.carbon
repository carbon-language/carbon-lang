// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/qualified_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/qualified_param.carbon

// --- qualifiers.h

void TakesConstInt(const int);

struct S {};

void TakesConstS(const S);

// --- call_qualifiers.carbon

library "[[@TEST_NAME]]";

import Cpp library "qualifiers.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.TakesConstInt(42);

  var s: Cpp.S = {};
  Cpp.TakesConstS(s);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- call_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesConstInt.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstInt.cpp_overload_set.value: %TakesConstInt.cpp_overload_set.type = cpp_overload_set_value @TakesConstInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %TakesConstInt.type: type = fn_type @TakesConstInt [concrete]
// CHECK:STDOUT:   %TakesConstInt: %TakesConstInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesConstS.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstS.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstS.cpp_overload_set.value: %TakesConstS.cpp_overload_set.type = cpp_overload_set_value @TakesConstS.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstS__carbon_thunk.type: type = fn_type @TakesConstS__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstS__carbon_thunk: %TakesConstS__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesConstInt = %TakesConstInt.cpp_overload_set.value
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .TakesConstS = %TakesConstS.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakesConstInt.cpp_overload_set.value: %TakesConstInt.cpp_overload_set.type = cpp_overload_set_value @TakesConstInt.cpp_overload_set [concrete = constants.%TakesConstInt.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesConstInt.decl: %TakesConstInt.type = fn_decl @TakesConstInt [concrete = constants.%TakesConstInt] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %TakesConstS.cpp_overload_set.value: %TakesConstS.cpp_overload_set.type = cpp_overload_set_value @TakesConstS.cpp_overload_set [concrete = constants.%TakesConstS.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesConstS__carbon_thunk.decl: %TakesConstS__carbon_thunk.type = fn_decl @TakesConstS__carbon_thunk [concrete = constants.%TakesConstS__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstInt.ref: %TakesConstInt.cpp_overload_set.type = name_ref TakesConstInt, imports.%TakesConstInt.cpp_overload_set.value [concrete = constants.%TakesConstInt.cpp_overload_set.value]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %TakesConstInt.call: init %empty_tuple.type = call imports.%TakesConstInt.decl(%_.var)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = ref_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc10_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_19.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc10_3: init %S = converted %.loc10_19.1, %.loc10_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc10_3
// CHECK:STDOUT:   %.loc10_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = ref_binding s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstS.ref: %TakesConstS.cpp_overload_set.type = name_ref TakesConstS, imports.%TakesConstS.cpp_overload_set.value [concrete = constants.%TakesConstS.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %.loc11_19.1: %S = acquire_value %s.ref
// CHECK:STDOUT:   %.loc11_19.2: ref %S = value_as_ref %.loc11_19.1
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %.loc11_19.2
// CHECK:STDOUT:   %.loc11_20.1: %ptr.ff5 = as_compatible %addr
// CHECK:STDOUT:   %.loc11_20.2: %ptr.ff5 = converted %addr, %.loc11_20.1
// CHECK:STDOUT:   %TakesConstS__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesConstS__carbon_thunk.decl(%.loc11_20.2)
// CHECK:STDOUT:   %S.cpp_destructor.bound: <bound method> = bound_method %s.var, constants.%S.cpp_destructor
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
