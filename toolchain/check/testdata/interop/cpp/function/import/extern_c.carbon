// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/extern_c.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/extern_c.carbon

// --- extern_c_function.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
extern "C" void foo_c_fn();
''';

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.foo_c_fn();
  //@dump-sem-ir-end
}

// --- extern_c_function_with_cpp_overload_set.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
extern "C" void foo_c_overloaded();
void foo_c_overloaded(int x);
''';

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.foo_c_overloaded();
  Cpp.foo_c_overloaded(5);
  //@dump-sem-ir-end
}

// --- extern_c_variable.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
extern "C" int foo_c_var;
''';

fn MyF() -> i32 {
  //@dump-sem-ir-begin
  return Cpp.foo_c_var;
  //@dump-sem-ir-end
}

// --- extern_c_with_special_name.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct X {};
extern "C" X operator+(X, X);
''';

fn MyF(a: Cpp.X, b: Cpp.X) -> Cpp.X {
  //@dump-sem-ir-begin
  return a + b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- extern_c_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_c_fn.cpp_overload_set.type: type = cpp_overload_set_type @foo_c_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_c_fn.cpp_overload_set.value: %foo_c_fn.cpp_overload_set.type = cpp_overload_set_value @foo_c_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_c_fn.type: type = fn_type @foo_c_fn [concrete]
// CHECK:STDOUT:   %foo_c_fn: %foo_c_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_c_fn = %foo_c_fn.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_c_fn.cpp_overload_set.value: %foo_c_fn.cpp_overload_set.type = cpp_overload_set_value @foo_c_fn.cpp_overload_set [concrete = constants.%foo_c_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_c_fn.decl: %foo_c_fn.type = fn_decl @foo_c_fn [concrete = constants.%foo_c_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_c_fn.ref: %foo_c_fn.cpp_overload_set.type = name_ref foo_c_fn, imports.%foo_c_fn.cpp_overload_set.value [concrete = constants.%foo_c_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_c_fn.call: init %empty_tuple.type = call imports.%foo_c_fn.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_c_function_with_cpp_overload_set.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_c_overloaded.cpp_overload_set.type: type = cpp_overload_set_type @foo_c_overloaded.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_c_overloaded.cpp_overload_set.value: %foo_c_overloaded.cpp_overload_set.type = cpp_overload_set_value @foo_c_overloaded.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_c_overloaded.type.7eb664.1: type = fn_type @foo_c_overloaded.1 [concrete]
// CHECK:STDOUT:   %foo_c_overloaded.fee3ce.1: %foo_c_overloaded.type.7eb664.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %foo_c_overloaded.type.7eb664.2: type = fn_type @foo_c_overloaded.2 [concrete]
// CHECK:STDOUT:   %foo_c_overloaded.fee3ce.2: %foo_c_overloaded.type.7eb664.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.eef: %i32 = int_value 5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_c_overloaded = %foo_c_overloaded.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_c_overloaded.cpp_overload_set.value: %foo_c_overloaded.cpp_overload_set.type = cpp_overload_set_value @foo_c_overloaded.cpp_overload_set [concrete = constants.%foo_c_overloaded.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_c_overloaded.decl.9ccf4c.1: %foo_c_overloaded.type.7eb664.1 = fn_decl @foo_c_overloaded.1 [concrete = constants.%foo_c_overloaded.fee3ce.1] {} {}
// CHECK:STDOUT:   %foo_c_overloaded.decl.9ccf4c.2: %foo_c_overloaded.type.7eb664.2 = fn_decl @foo_c_overloaded.2 [concrete = constants.%foo_c_overloaded.fee3ce.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_c_overloaded.ref.loc11: %foo_c_overloaded.cpp_overload_set.type = name_ref foo_c_overloaded, imports.%foo_c_overloaded.cpp_overload_set.value [concrete = constants.%foo_c_overloaded.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_c_overloaded.call.loc11: init %empty_tuple.type = call imports.%foo_c_overloaded.decl.9ccf4c.1()
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_c_overloaded.ref.loc12: %foo_c_overloaded.cpp_overload_set.type = name_ref foo_c_overloaded, imports.%foo_c_overloaded.cpp_overload_set.value [concrete = constants.%foo_c_overloaded.cpp_overload_set.value]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc12_24.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_24.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_24.2(%int_5) [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %.loc12_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %.loc12_24.2: %i32 = converted %int_5, %.loc12_24.1 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %foo_c_overloaded.call.loc12: init %empty_tuple.type = call imports.%foo_c_overloaded.decl.9ccf4c.2(%.loc12_24.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_c_variable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %foo_c_var.patt: %pattern_type.6b6 = ref_binding_pattern foo_c_var [concrete]
// CHECK:STDOUT:   %foo_c_var.var_patt: %pattern_type.6b6 = var_pattern %foo_c_var.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_c_var = %foo_c_var.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_c_var.patt: %pattern_type.6b6 = ref_binding_pattern foo_c_var [concrete = constants.%foo_c_var.patt]
// CHECK:STDOUT:   %foo_c_var.var_patt: %pattern_type.6b6 = var_pattern %foo_c_var.patt [concrete = constants.%foo_c_var.var_patt]
// CHECK:STDOUT:   %foo_c_var.var: ref %i32 = var_storage %foo_c_var.var_patt [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %foo_c_var.ref: ref %i32 = name_ref foo_c_var, imports.%foo_c_var.var [concrete = imports.%foo_c_var.var]
// CHECK:STDOUT:   %.loc10: %i32 = acquire_value %foo_c_var.ref
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc10_13.1: <bound method> = bound_method %.loc10, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_13.2: <bound method> = bound_method %.loc10, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_13.2(%.loc10)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_c_with_special_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.ca1: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.type: type = fn_type @operator_Plus__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_Plus__carbon_thunk: %operator_Plus__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.decl: %operator_Plus__carbon_thunk.type = fn_decl @operator_Plus__carbon_thunk [concrete = constants.%operator_Plus__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%a.param: %X, %b.param: %X) -> out %return.param: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %X = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %X = name_ref b, %b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc11_10: ref %X = value_as_ref %a.ref
// CHECK:STDOUT:   %addr.loc11_12.1: %ptr.ca1 = addr_of %.loc11_10
// CHECK:STDOUT:   %.loc11_14: ref %X = value_as_ref %b.ref
// CHECK:STDOUT:   %addr.loc11_12.2: %ptr.ca1 = addr_of %.loc11_14
// CHECK:STDOUT:   %addr.loc11_12.3: %ptr.ca1 = addr_of %.loc9_34.1
// CHECK:STDOUT:   %operator_Plus__carbon_thunk.call: init %empty_tuple.type = call imports.%operator_Plus__carbon_thunk.decl(%addr.loc11_12.1, %addr.loc11_12.2, %addr.loc11_12.3)
// CHECK:STDOUT:   %.loc11_12: init %X to %.loc9_34.1 = mark_in_place_init %operator_Plus__carbon_thunk.call
// CHECK:STDOUT:   return %.loc11_12 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
