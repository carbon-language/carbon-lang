// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --clang-arg=--std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/constexpr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/constexpr.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int add(int a, int b) { return a + b; }
''';

let a: array(i32, Cpp.add(1, 2)) = (1, 2, 3);

// --- bool_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int bool_fn(bool b) {
  return b ? 3 : 0;
}
''';

let a: array(i32, Cpp.bool_fn(true)) = (1, 2, 3);

// --- float_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int float_fn(float b) {
  return static_cast<int>(b);
}
''';

let a: array(i32, Cpp.float_fn(3.0)) = (1, 2, 3);

// --- return_bool.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr bool return_bool_fn() {
  return true;
}
''';

musteval fn F(b: bool) -> i32 {
  return if b then 3 else 0;
}

let a: array(i32, F(Cpp.return_bool_fn())) = (1, 2, 3);

// --- fail_invalid_constant_eval.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int arr[1] = {0};
constexpr int invalid_const_fn(int a) {
  return arr[a];
}
''';

// CHECK:STDERR: fail_invalid_constant_eval.carbon:[[@LINE+2]]:19-41: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR:
let a: array(i32, Cpp.invalid_const_fn(5)) = (1, 2, 3);

// --- nonconstant_constexpr_call.carbon

library "[[@TEST_NAME]]";

import Cpp;

var a: i32 = 0;

inline Cpp '''
constexpr int nonconst_constexpr_fn(const int &r) { return r; }
''';

// OK, runtime call.
var b: i32 = Cpp.nonconst_constexpr_fn(a);

// --- fail_nonconstant_consteval_call.carbon

library "[[@TEST_NAME]]";

import Cpp;

var a: i32 = 0;

// TODO: This is failing when building the thunk, not when calling it.
inline Cpp '''
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+3]]:15: error: call to consteval function 'nonconst_consteval_fn' is not a constant expression [CppInteropParseError]
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+2]]:15: note: function parameter 'r' with unknown value cannot be used in a constant expression [CppInteropParseNote]
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+1]]:15: note: declared here [CppInteropParseNote]
consteval int nonconst_consteval_fn(const int &r) { return r; }
''';

// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+5]]:14-41: note: in thunk for C++ function used here [InCppThunk]
// CHECK:STDERR:
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE+3]]:14-41: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: fail_nonconstant_consteval_call.carbon:[[@LINE-6]]:15: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR:
var b: i32 = Cpp.nonconst_consteval_fn(a);

// --- fail_todo_constructor.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
struct ConstexprC {
  constexpr ConstexprC(int a) : a(a) {}
  int a;
};
''';

// TODO: This should probably be allowed.
eval fn F() -> i32 {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_constructor.carbon:[[@LINE+1]]:27: error: expression is runtime; expected constant [EvalRequiresConstantValue]
  let c: Cpp.ConstexprC = 3;
  return c.a;
  //@dump-sem-ir-end
}

// CHECK:STDERR: fail_todo_constructor.carbon:[[@LINE+2]]:19-21: note: in call to F here [InCallToEvalFn]
// CHECK:STDERR:
let a: array(i32, F()) = (1, 2, 3);

// CHECK:STDOUT: --- fail_todo_constructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ConstexprC: type = class_type @ConstexprC [concrete]
// CHECK:STDOUT:   %ConstexprC.elem: type = unbound_element_type %ConstexprC, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.3cc: type = pattern_type %ConstexprC [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.3cc = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %ptr.0b8: type = ptr_type %ConstexprC [concrete]
// CHECK:STDOUT:   %ConstexprC__carbon_thunk.type: type = fn_type @ConstexprC__carbon_thunk [concrete]
// CHECK:STDOUT:   %ConstexprC__carbon_thunk: %ConstexprC__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.883: %SubobjectDestroy.type = facet_value %ConstexprC, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.025: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.883) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ConstexprC, (%DestroyImpl.impl_witness.025, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ConstexprC = %ConstexprC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstexprC.decl: type = class_decl @ConstexprC [concrete = constants.%ConstexprC] {} {}
// CHECK:STDOUT:   %ConstexprC__carbon_thunk.decl: %ConstexprC__carbon_thunk.type = fn_decl @ConstexprC__carbon_thunk [concrete = constants.%ConstexprC__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc17_27.1: ref %ConstexprC = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc17: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc17_27.1: <bound method> = bound_method %int_3, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_27.2: <bound method> = bound_method %int_3, %specific_fn.loc17 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc17_27.2(%int_3) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc17_27.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc17_27.3: %i32 = converted %int_3, %.loc17_27.2 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %addr: %ptr.0b8 = addr_of %.loc17_27.1
// CHECK:STDOUT:   %ConstexprC__carbon_thunk.call: init %empty_tuple.type = call imports.%ConstexprC__carbon_thunk.decl(%.loc17_27.3, %addr)
// CHECK:STDOUT:   %.loc17_27.4: init %ConstexprC to %.loc17_27.1 = mark_in_place_init %ConstexprC__carbon_thunk.call
// CHECK:STDOUT:   %.loc17_27.5: init %ConstexprC = converted %int_3, %.loc17_27.4
// CHECK:STDOUT:   %.loc17_27.6: ref %ConstexprC = temporary %.loc17_27.1, %.loc17_27.5
// CHECK:STDOUT:   %.loc17_27.7: %ConstexprC = acquire_value %.loc17_27.6
// CHECK:STDOUT:   %.loc17_13: type = splice_block %ConstexprC.ref [concrete = constants.%ConstexprC] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ConstexprC.ref: type = name_ref ConstexprC, imports.%ConstexprC.decl [concrete = constants.%ConstexprC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %ConstexprC = wrapper_binding c, %.loc17_27.7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.3cc = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstexprC = name_ref c, %c
// CHECK:STDOUT:   %a.ref: %ConstexprC.elem = name_ref a, @ConstexprC.%field_decl [concrete = @ConstexprC.%field_decl]
// CHECK:STDOUT:   %.loc18_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc18_11.2: %i32 = acquire_value %.loc18_11.1
// CHECK:STDOUT:   %impl.elem0.loc18: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc18_11.1: <bound method> = bound_method %.loc18_11.2, %impl.elem0.loc18
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_11.2: <bound method> = bound_method %.loc18_11.2, %specific_fn.loc18
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc18_11.2(%.loc18_11.2)
// CHECK:STDOUT:   %Destroy.facet.loc17_27.1: %Destroy.type = facet_value constants.%ConstexprC, (constants.%DestroyImpl.impl_witness.025, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_27.8: %Destroy.type = converted constants.%ConstexprC, %Destroy.facet.loc17_27.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc17_27.2: %Destroy.type = facet_value constants.%ConstexprC, (constants.%DestroyImpl.impl_witness.025, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc17_27.9: %Destroy.type = converted constants.%ConstexprC, %Destroy.facet.loc17_27.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc17_27.6)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17(%self.param: ref %ConstexprC) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
