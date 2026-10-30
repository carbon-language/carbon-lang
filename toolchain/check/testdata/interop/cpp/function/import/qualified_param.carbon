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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesConstInt.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstInt.cpp_overload_set.value: %TakesConstInt.cpp_overload_set.type = cpp_overload_set_value @TakesConstInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %TakesConstInt.type: type = fn_type @TakesConstInt [concrete]
// CHECK:STDOUT:   %TakesConstInt: %TakesConstInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %TakesConstS.cpp_overload_set.type: type = cpp_overload_set_type @TakesConstS.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesConstS.cpp_overload_set.value: %TakesConstS.cpp_overload_set.type = cpp_overload_set_value @TakesConstS.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr.711: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %TakesConstS__carbon_thunk.type: type = fn_type @TakesConstS__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakesConstS__carbon_thunk: %TakesConstS__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.261: %SubobjectDestroy.type = facet_value %S, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8fa: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.261) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %S, (%DestroyImpl.impl_witness.8fa, %custom_witness.b80) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %TakesConstS.cpp_overload_set.value: %TakesConstS.cpp_overload_set.type = cpp_overload_set_value @TakesConstS.cpp_overload_set [concrete = constants.%TakesConstS.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesConstS__carbon_thunk.decl: %TakesConstS__carbon_thunk.type = fn_decl @TakesConstS__carbon_thunk [concrete = constants.%TakesConstS__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstInt.ref: %TakesConstInt.cpp_overload_set.type = name_ref TakesConstInt, imports.%TakesConstInt.cpp_overload_set.value [concrete = constants.%TakesConstInt.cpp_overload_set.value]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_21.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_21.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_21.2(%int_42) [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc8_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc8_21.2: %i32 = converted %int_42, %.loc8_21.1 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %TakesConstInt.call: init %empty_tuple.type = call imports.%TakesConstInt.decl(%.loc8_21.2)
// CHECK:STDOUT:   %s.var: ref %S = var_storage %s.var_patt
// CHECK:STDOUT:   %.loc10_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_19.2: init %S to %s.var = class_init () [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc10_3.1: init %S = converted %.loc10_19.1, %.loc10_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc10_3.1
// CHECK:STDOUT:   %.loc10_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = wrapper_binding s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.0f5 = ref_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.0f5 = var_pattern %s.patt [concrete = constants.%s.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesConstS.ref: %TakesConstS.cpp_overload_set.type = name_ref TakesConstS, imports.%TakesConstS.cpp_overload_set.value [concrete = constants.%TakesConstS.cpp_overload_set.value]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %.loc11_19.1: %S = acquire_value %s.ref
// CHECK:STDOUT:   %.loc11_19.2: ref %S = value_as_ref %.loc11_19.1
// CHECK:STDOUT:   %addr: %ptr.037 = addr_of %.loc11_19.2
// CHECK:STDOUT:   %.loc11_20.1: %ptr.711 = as_compatible %addr
// CHECK:STDOUT:   %.loc11_20.2: %ptr.711 = converted %addr, %.loc11_20.1
// CHECK:STDOUT:   %TakesConstS__carbon_thunk.call: init %empty_tuple.type = call imports.%TakesConstS__carbon_thunk.decl(%.loc11_20.2)
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%S, (constants.%DestroyImpl.impl_witness.8fa, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%S, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%S, (constants.%DestroyImpl.impl_witness.8fa, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%S, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%s.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %S) = "no_op";
// CHECK:STDOUT:
