// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/template.carbon

// --- class_template.h

struct Base {};

template<typename T> struct A : Base {
  void f() const;
  T n;
};

using Ai32 = A<int>;
using Ai16 = A<short>;

// --- use_class_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_template.h";

fn F(x: Cpp.Ai32) -> i32 {
  // Trigger instantiation through name lookup.
  //@dump-sem-ir-begin
  x.f();
  return x.n;
  //@dump-sem-ir-end
}

fn G(p: Cpp.Ai16*) -> Cpp.Base* {
  // Trigger instantiation through type completion.
  //@dump-sem-ir-begin
  return p;
  //@dump-sem-ir-end
}

// --- instantiation_error.h

template<typename T> struct X {
  using type = typename T::member;
};

struct Y {
  using member = int;
};

using XY = X<Y>;
using Xint = X<int>;

// --- fail_use_instantiation_error.carbon

library "[[@TEST_NAME]]";

import Cpp library "instantiation_error.h";

//@dump-sem-ir-begin
var x: Cpp.XY.r#type = 0;

// CHECK:STDERR: fail_use_instantiation_error.carbon:[[@LINE+5]]:8-22: error: member access into incomplete class `Cpp.X` [QualifiedExprInIncompleteClassScope]
// CHECK:STDERR: included from: fail_use_instantiation_error.carbon:[[@LINE-6]]:10 [InCppInclude]
// CHECK:STDERR: ./instantiation_error.h:3:25: note: type 'int' cannot be used prior to '::' because it has no members [CppInteropParseError]
// CHECK:STDERR: ./instantiation_error.h:2:29: note: in instantiation of template class 'X<int>' requested here [CppInteropParseNote]
// CHECK:STDERR:
var y: Cpp.Xint.r#type = 0;
//@dump-sem-ir-end

// --- use_indirect_class_template.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
template<typename T> struct wrap_iter {};
template<typename T> struct vector {
  using iterator = wrap_iter<T*>;
  iterator begin();
};
''';

fn F() {
  var v: Cpp.vector(i32);
  v.begin();
}

// --- use_carbon_class_generic_in_cpp_template_param.carbon

library "[[@TEST_NAME]]";
import Cpp inline '''
template<typename T>
struct C {
  T t;
};
''';

//@dump-sem-ir-begin
class B(T: type) {
  var a: Cpp.C(T);
}

fn F(ref b: B(i32)) {
  b.a.t = 123;
}
//@dump-sem-ir-end

// --- let_cpp_template_constructor.carbon

library "[[@TEST_NAME]]";
import Cpp inline '''
template<typename T>
struct D {};
''';

//@dump-sem-ir-begin
fn F(generic T: type) {
  let unused a: Cpp.D(T) = Cpp.D(T).D();
}

fn G() {
  F(i32);
}
//@dump-sem-ir-end

// --- var_cpp_template_constructor.carbon

library "[[@TEST_NAME]]";
import Cpp inline '''
template<typename T>
struct E {};
''';

//@dump-sem-ir-begin
fn F(generic T: type) {
  var unused a: Cpp.E(T) = Cpp.E(T).E();
}
//@dump-sem-ir-end

fn G() {
  F(i32);
}

// CHECK:STDOUT: --- use_class_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A.614: type = class_type @A.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %A.elem.c38: type = unbound_element_type %A.614, %i32 [concrete]
// CHECK:STDOUT:   %A.f.cpp_overload_set.type: type = cpp_overload_set_type @A.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %A.f.cpp_overload_set.value: %A.f.cpp_overload_set.type = cpp_overload_set_value @A.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %f__carbon_thunk.type: type = fn_type @f__carbon_thunk [concrete]
// CHECK:STDOUT:   %f__carbon_thunk: %f__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %A.e3f: type = class_type @A.2 [concrete]
// CHECK:STDOUT:   %ptr.bc0: type = ptr_type %A.e3f [concrete]
// CHECK:STDOUT:   %ptr.0e0: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.39c: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.903: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.982: %ptr.as.Copy.impl.Op.type.903 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.ee1: %Copy.type = facet_value %ptr.0e0, (%Copy.impl_witness.39c) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e2a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ee1) [concrete]
// CHECK:STDOUT:   %.2aa: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e2a, %Copy.facet.ee1 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.982, @ptr.as.Copy.impl.Op(%Base) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %A.f.cpp_overload_set.value: %A.f.cpp_overload_set.type = cpp_overload_set_value @A.f.cpp_overload_set [concrete = constants.%A.f.cpp_overload_set.value]
// CHECK:STDOUT:   %f__carbon_thunk.decl: %f__carbon_thunk.type = fn_decl @f__carbon_thunk [concrete = constants.%f__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %A.614) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc9: %A.614 = name_ref x, %x
// CHECK:STDOUT:   %f.ref: %A.f.cpp_overload_set.type = name_ref f, imports.%A.f.cpp_overload_set.value [concrete = constants.%A.f.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %x.ref.loc9, %f.ref
// CHECK:STDOUT:   %f__carbon_thunk.call: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%x.ref.loc9)
// CHECK:STDOUT:   %x.ref.loc10: %A.614 = name_ref x, %x
// CHECK:STDOUT:   %n.ref: %A.elem.c38 = name_ref n, @A.1.%field_decl [concrete = @A.1.%field_decl]
// CHECK:STDOUT:   %.loc10_11.1: ref %i32 = class_element_access %x.ref.loc10, element1
// CHECK:STDOUT:   %.loc10_11.2: %i32 = acquire_value %.loc10_11.1
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.bc0) -> out %return.param: %ptr.0e0 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.bc0 = name_ref p, %p
// CHECK:STDOUT:   %.loc17_11.1: ref %A.e3f = deref %p.ref
// CHECK:STDOUT:   %.loc17_11.2: ref %Base = class_element_access %.loc17_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.0e0 = addr_of %.loc17_11.2
// CHECK:STDOUT:   %.loc17_11.3: %ptr.0e0 = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.2aa = impl_witness_access constants.%Copy.impl_witness.39c, element0 [concrete = constants.%ptr.as.Copy.impl.Op.982]
// CHECK:STDOUT:   %bound_method.loc17_11.1: <bound method> = bound_method %.loc17_11.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Base) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_11.2: <bound method> = bound_method %.loc17_11.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.0e0 = call %bound_method.loc17_11.2(%.loc17_11.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_instantiation_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X.816: type = class_type @X.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %X.c92: type = class_type @X.2 [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.d94 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.d94 = var_pattern %y.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .XY = %X.decl.8c4
// CHECK:STDOUT:     .Xint = %X.decl.b67
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl.8c4: type = class_decl @X.1 [concrete = constants.%X.816] {} {}
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X.decl.b67: type = class_decl @X.2 [concrete = constants.%X.c92] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %type.ref.loc7 [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %XY.ref: type = name_ref XY, imports.%X.decl.8c4 [concrete = constants.%X.816]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %type.ref.loc7: type = name_ref r#type, %i32.4 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %i32 = wrapper_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var_storage %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc14: type = splice_block %type.ref.loc14 [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Xint.ref: type = name_ref Xint, imports.%X.decl.b67 [concrete = constants.%X.c92]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %type.ref.loc14: type = name_ref r#type, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %i32 = wrapper_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.d94 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.d94 = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc7: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc7: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_1.1: <bound method> = bound_method %int_0.loc7, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_1.2: <bound method> = bound_method %int_0.loc7, %specific_fn.loc7 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_1.2(%int_0.loc7) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc7: init %i32 = converted %int_0.loc7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   assign file.%x.var, %.loc7
// CHECK:STDOUT:   %int_0.loc14: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc14: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc14_1.1: <bound method> = bound_method %int_0.loc14, %impl.elem0.loc14 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_1.2: <bound method> = bound_method %int_0.loc14, %specific_fn.loc14 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_1.2(%int_0.loc14) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc14: init %i32 = converted %int_0.loc14, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   assign file.%y.var, %.loc14
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_carbon_class_generic_in_cpp_template_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = parameterized_class_name_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.6a4: type = class_type @B, @B(%T) [symbolic]
// CHECK:STDOUT:   %C.type: type = cpp_type_template_type C [concrete]
// CHECK:STDOUT:   %C.template: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a92: type = template_inst %T [template]
// CHECK:STDOUT:   %.36a: type = type_of_inst @B.%.loc12_17.4 [template]
// CHECK:STDOUT:   %.81b: %.36a = splice_inst @B.%.loc12_17.4 [template]
// CHECK:STDOUT:   %.537: type = splice_inst @B.%.loc12_17.7 [template]
// CHECK:STDOUT:   %require_complete.cb0: <witness> = require_complete_type %.537 [template]
// CHECK:STDOUT:   %B.elem.930: type = unbound_element_type %B.6a4, %.537 [template]
// CHECK:STDOUT:   %struct_type.a.df5: type = struct_type {.a: %.537} [template]
// CHECK:STDOUT:   %complete_type.e6d: <witness> = complete_type_witness %struct_type.a.df5 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.6f4: type = class_type @B, @B(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.d2f: type = pattern_type %B.6f4 [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.d2f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d2f = wrapper_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.792: type = splice_block imports.%C.decl [concrete = %C] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.3e0: type = splice_block %C [concrete = %C] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %.0fe: type = custom_layout_type {size=4, align=4, .t@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.4fc: <witness> = complete_type_witness %.0fe [concrete]
// CHECK:STDOUT:   %B.elem.409: type = unbound_element_type %B.6f4, %C [concrete]
// CHECK:STDOUT:   %struct_type.a.9d2: type = struct_type {.a: %C} [concrete]
// CHECK:STDOUT:   %complete_type.12b: <witness> = complete_type_witness %struct_type.a.9d2 [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.c07: %i32 = int_value 123 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.template: %C.type = struct_value () [concrete = constants.%C.template]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %B.decl: %B.type = class_decl @B [concrete = constants.%B.generic] {
// CHECK:STDOUT:     %T.patt.loc11_10.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: type = splice_block %.loc11_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc11_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.d2f = ref_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.d2f = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: ref %B.6f4 = ref_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %B [concrete = constants.%B.6f4] {
// CHECK:STDOUT:       %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %B: type = class_type @B, @B(constants.%i32) [concrete = constants.%B.6f4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: ref %B.6f4 = wrapper_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B(%T.loc11_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc11_10.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc11_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc12_16.2: type = template_inst %T.loc11_10.1 [template = %.loc12_16.2 (constants.%.a92)]
// CHECK:STDOUT:   %.loc12_17.4: <instruction> = call_action (%C.ref, %.loc12_16.1), false [template]
// CHECK:STDOUT:   %.loc12_17.5: type = type_of_inst %.loc12_17.4 [template = %.loc12_17.5 (constants.%.36a)]
// CHECK:STDOUT:   %.loc12_17.6: @B.%.loc12_17.5 (%.36a) = splice_inst %.loc12_17.4 [template = %.loc12_17.6 (constants.%.81b)]
// CHECK:STDOUT:   %.loc12_17.7: <instruction> = convert_to_value_action %.loc12_17.2, type [template]
// CHECK:STDOUT:   %.loc12_17.8: type = splice_inst %.loc12_17.7 [template = %.loc12_17.8 (constants.%.537)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc12_17.8 [template = %require_complete (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T.loc11_10.1) [symbolic = %B (constants.%B.6a4)]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %.loc12_17.8 [template = %B.elem (constants.%B.elem.930)]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: @B.%.loc12_17.8 (%.537)} [template = %struct_type.a (constants.%struct_type.a.df5)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness %struct_type.a [template = %complete_type.loc13_1.2 (constants.%complete_type.e6d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %field_decl: @B.%B.elem (%B.elem.930) = field_decl a, element0, %.loc12_17.3 in [concrete] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%C.template [concrete = constants.%C.template]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc11_10.2 [symbolic = %T.loc11_10.1 (constants.%T)]
// CHECK:STDOUT:       %.loc12_16.1: type = template_inst %T.ref [template = %.loc12_16.2 (constants.%.a92)]
// CHECK:STDOUT:       %.loc12_17.1: type = type_of_inst %.loc12_17.4 [template = %.loc12_17.5 (constants.%.36a)]
// CHECK:STDOUT:       %.loc12_17.2: @B.%.loc12_17.5 (%.36a) = splice_inst %.loc12_17.4 [template = %.loc12_17.6 (constants.%.81b)]
// CHECK:STDOUT:       %.loc12_17.3: type = splice_inst %.loc12_17.7 [template = %.loc12_17.8 (constants.%.537)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness constants.%struct_type.a.df5 [template = %complete_type.loc13_1.2 (constants.%complete_type.e6d)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B.6a4
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .a = %field_decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: ref %B.6f4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: ref %B.6f4 = name_ref b, %b
// CHECK:STDOUT:   %a.ref: %B.elem.409 = name_ref a, @B.%field_decl [concrete = @B.%field_decl]
// CHECK:STDOUT:   %.loc16_4: ref %C = class_element_access %b.ref, element0
// CHECK:STDOUT:   %t.ref: %C.elem = name_ref t, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:   %.loc16_6: ref %i32 = class_element_access %.loc16_4, element0
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc16_9.1: <bound method> = bound_method %int_123, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_9.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc16_9.2(%int_123) [concrete = constants.%int_123.c07]
// CHECK:STDOUT:   %.loc16_9: init %i32 = converted %int_123, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.c07]
// CHECK:STDOUT:   assign %.loc16_6, %.loc16_9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc11_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc11_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc11_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc11_10.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc12_16.2 => constants.%i32
// CHECK:STDOUT:   %.loc12_17.4 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc12_17.5 => type
// CHECK:STDOUT:   %.loc12_17.6 => constants.%C
// CHECK:STDOUT:   %.loc12_17.7 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc12_17.8 => constants.%C
// CHECK:STDOUT:   %require_complete => constants.%complete_type.4fc
// CHECK:STDOUT:   %B => constants.%B.6f4
// CHECK:STDOUT:   %B.elem => constants.%B.elem.409
// CHECK:STDOUT:   %struct_type.a => constants.%struct_type.a.9d2
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type.12b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- let_cpp_template_constructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = cpp_type_template_type D [concrete]
// CHECK:STDOUT:   %D.template: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a92: type = template_inst %T [template]
// CHECK:STDOUT:   %.d03c60.1: type = type_of_inst @F.%.loc10_24.5 [template]
// CHECK:STDOUT:   %.9262b2.1: %.d03c60.1 = splice_inst @F.%.loc10_24.5 [template]
// CHECK:STDOUT:   %.537: type = splice_inst @F.%.loc10_24.8 [template]
// CHECK:STDOUT:   %require_complete.cb0: <witness> = require_complete_type %.537 [template]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %.537 [template]
// CHECK:STDOUT:   %a.patt.d9b: %pattern_type.fcb = value_binding_pattern a [template]
// CHECK:STDOUT:   %.d03c60.2: type = type_of_inst @F.%.loc10_35.3 [template]
// CHECK:STDOUT:   %.9262b2.2: %.d03c60.2 = splice_inst @F.%.loc10_35.3 [template]
// CHECK:STDOUT:   %.588: type = type_of_inst @F.%.loc10_36.3 [template]
// CHECK:STDOUT:   %.f78: %.588 = splice_inst @F.%.loc10_36.3 [template]
// CHECK:STDOUT:   %.e26: type = type_of_inst @F.%.loc10_39.4 [template]
// CHECK:STDOUT:   %.60e: %.e26 = splice_inst @F.%.loc10_39.4 [template]
// CHECK:STDOUT:   %.739: %.537 = splice_inst @F.%.loc10_39.7 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%i32) [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.0fe0af.1: type = splice_block imports.%D.decl [concrete = %D] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b94: type = splice_block %D [concrete = %D] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.af5: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %a.patt.2c7: %pattern_type.af5 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.0fe0af.2: type = splice_block imports.%D.decl [concrete = %D] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.D.cpp_overload_set.type: type = cpp_overload_set_type @D.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %D.D.cpp_overload_set.value: %D.D.cpp_overload_set.type = cpp_overload_set_value @D.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %inst.name_ref: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %D.ref: %D.D.cpp_overload_set.type = name_ref D, imports.%D.D.cpp_overload_set.value [concrete = %D.D.cpp_overload_set.value]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.342: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %D__carbon_thunk.type: type = fn_type @D__carbon_thunk [concrete]
// CHECK:STDOUT:   %D__carbon_thunk: %D__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.6a7: init %D = splice_block %.d90 {
// CHECK:STDOUT:       %.f50: ref %D = temporary_storage
// CHECK:STDOUT:       %addr: %ptr.342 = addr_of %.f50
// CHECK:STDOUT:       %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr)
// CHECK:STDOUT:       %.d90: init %D to %.f50 = mark_in_place_init %D__carbon_thunk.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.0ee: %D = splice_block %.2f1 {
// CHECK:STDOUT:       %.715: ref %D = temporary %.f50, %.6a7
// CHECK:STDOUT:       %.2f1: %D = acquire_value %.715
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .D = %D.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.template: %D.type = struct_value () [concrete = constants.%D.template]
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %D.D.cpp_overload_set.value: %D.D.cpp_overload_set.type = cpp_overload_set_value @D.D.cpp_overload_set [concrete = constants.%D.D.cpp_overload_set.value]
// CHECK:STDOUT:   %D__carbon_thunk.decl: %D__carbon_thunk.type = fn_decl @D__carbon_thunk [concrete = constants.%D__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_15.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_15.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc9_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc10_23.2: type = template_inst %T.loc9_15.1 [template = %.loc10_23.2 (constants.%.a92)]
// CHECK:STDOUT:   %.loc10_24.5: <instruction> = call_action (%D.ref.loc10_20, %.loc10_23.1), false [template]
// CHECK:STDOUT:   %.loc10_24.6: type = type_of_inst %.loc10_24.5 [template = %.loc10_24.6 (constants.%.d03c60.1)]
// CHECK:STDOUT:   %.loc10_24.7: @F.%.loc10_24.6 (%.d03c60.1) = splice_inst %.loc10_24.5 [template = %.loc10_24.7 (constants.%.9262b2.1)]
// CHECK:STDOUT:   %.loc10_24.8: <instruction> = convert_to_value_action %.loc10_24.3, type [template]
// CHECK:STDOUT:   %.loc10_24.9: type = splice_inst %.loc10_24.8 [template = %.loc10_24.9 (constants.%.537)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc10_24.9 [template = %require_complete (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc10_24.9 [template = %pattern_type (constants.%pattern_type.fcb)]
// CHECK:STDOUT:   %a.patt.loc10_15.2: @F.%pattern_type (%pattern_type.fcb) = value_binding_pattern a [template = %a.patt.loc10_15.2 (constants.%a.patt.d9b)]
// CHECK:STDOUT:   %.loc10_35.3: <instruction> = call_action (%D.ref.loc10_31, %.loc10_34), false [template]
// CHECK:STDOUT:   %.loc10_35.4: type = type_of_inst %.loc10_35.3 [template = %.loc10_35.4 (constants.%.d03c60.2)]
// CHECK:STDOUT:   %.loc10_35.5: @F.%.loc10_35.4 (%.d03c60.2) = splice_inst %.loc10_35.3 [template = %.loc10_35.5 (constants.%.9262b2.2)]
// CHECK:STDOUT:   %.loc10_36.3: <instruction> = access_member_action %.loc10_35.2, D [template]
// CHECK:STDOUT:   %.loc10_36.4: type = type_of_inst %.loc10_36.3 [template = %.loc10_36.4 (constants.%.588)]
// CHECK:STDOUT:   %.loc10_36.5: @F.%.loc10_36.4 (%.588) = splice_inst %.loc10_36.3 [template = %.loc10_36.5 (constants.%.f78)]
// CHECK:STDOUT:   %.loc10_39.4: <instruction> = call_action (%.loc10_36.2), false [template]
// CHECK:STDOUT:   %.loc10_39.5: type = type_of_inst %.loc10_39.4 [template = %.loc10_39.5 (constants.%.e26)]
// CHECK:STDOUT:   %.loc10_39.6: @F.%.loc10_39.5 (%.e26) = splice_inst %.loc10_39.4 [template = %.loc10_39.6 (constants.%.60e)]
// CHECK:STDOUT:   %.loc10_39.7: <instruction> = convert_to_value_action %.loc10_39.2, %.loc10_24.9 [template]
// CHECK:STDOUT:   %.loc10_39.8: @F.%.loc10_24.9 (%.537) = splice_inst %.loc10_39.7 [template = %.loc10_39.8 (constants.%.739)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Cpp.ref.loc10_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %D.ref.loc10_31: %D.type = name_ref D, imports.%D.template [concrete = constants.%D.template]
// CHECK:STDOUT:     %T.ref.loc10_34: type = name_ref T, %T.loc9_15.2 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10_34: type = template_inst %T.ref.loc10_34 [template = %.loc10_23.2 (constants.%.a92)]
// CHECK:STDOUT:     %.loc10_35.1: type = type_of_inst %.loc10_35.3 [template = %.loc10_35.4 (constants.%.d03c60.2)]
// CHECK:STDOUT:     %.loc10_35.2: @F.%.loc10_35.4 (%.d03c60.2) = splice_inst %.loc10_35.3 [template = %.loc10_35.5 (constants.%.9262b2.2)]
// CHECK:STDOUT:     %.loc10_36.1: type = type_of_inst %.loc10_36.3 [template = %.loc10_36.4 (constants.%.588)]
// CHECK:STDOUT:     %.loc10_36.2: @F.%.loc10_36.4 (%.588) = splice_inst %.loc10_36.3 [template = %.loc10_36.5 (constants.%.f78)]
// CHECK:STDOUT:     %.loc10_39.1: type = type_of_inst %.loc10_39.4 [template = %.loc10_39.5 (constants.%.e26)]
// CHECK:STDOUT:     %.loc10_39.2: @F.%.loc10_39.5 (%.e26) = splice_inst %.loc10_39.4 [template = %.loc10_39.6 (constants.%.60e)]
// CHECK:STDOUT:     %.loc10_39.3: @F.%.loc10_24.9 (%.537) = splice_inst %.loc10_39.7 [template = %.loc10_39.8 (constants.%.739)]
// CHECK:STDOUT:     %.loc10_24.1: type = splice_block %.loc10_24.4 [template = %.loc10_24.9 (constants.%.537)] {
// CHECK:STDOUT:       %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %D.ref.loc10_20: %D.type = name_ref D, imports.%D.template [concrete = constants.%D.template]
// CHECK:STDOUT:       %T.ref.loc10_23: type = name_ref T, %T.loc9_15.2 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:       %.loc10_23.1: type = template_inst %T.ref.loc10_23 [template = %.loc10_23.2 (constants.%.a92)]
// CHECK:STDOUT:       %.loc10_24.2: type = type_of_inst %.loc10_24.5 [template = %.loc10_24.6 (constants.%.d03c60.1)]
// CHECK:STDOUT:       %.loc10_24.3: @F.%.loc10_24.6 (%.d03c60.1) = splice_inst %.loc10_24.5 [template = %.loc10_24.7 (constants.%.9262b2.1)]
// CHECK:STDOUT:       %.loc10_24.4: type = splice_inst %.loc10_24.8 [template = %.loc10_24.9 (constants.%.537)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc10_24.9 (%.537) = wrapper_binding a, %.loc10_39.3
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc10_15.1: @F.%pattern_type (%pattern_type.fcb) = value_binding_pattern a [template = %a.patt.loc10_15.2 (constants.%a.patt.d9b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%i32) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc9_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc9_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_15.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc10_23.2 => constants.%i32
// CHECK:STDOUT:   %.loc10_24.5 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc10_24.6 => type
// CHECK:STDOUT:   %.loc10_24.7 => constants.%D
// CHECK:STDOUT:   %.loc10_24.8 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc10_24.9 => constants.%D
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.af5
// CHECK:STDOUT:   %a.patt.loc10_15.2 => constants.%a.patt.2c7
// CHECK:STDOUT:   %.loc10_35.3 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc10_35.4 => type
// CHECK:STDOUT:   %.loc10_35.5 => constants.%D
// CHECK:STDOUT:   %.loc10_36.3 => constants.%inst.name_ref
// CHECK:STDOUT:   %.loc10_36.4 => constants.%D.D.cpp_overload_set.type
// CHECK:STDOUT:   %.loc10_36.5 => constants.%D.D.cpp_overload_set.value
// CHECK:STDOUT:   %.loc10_39.4 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc10_39.5 => constants.%D
// CHECK:STDOUT:   %.loc10_39.6 => <not constant>
// CHECK:STDOUT:   %.loc10_39.7 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc10_39.8 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_cpp_template_constructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = cpp_type_template_type E [concrete]
// CHECK:STDOUT:   %E.template: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a92: type = template_inst %T.67d [template]
// CHECK:STDOUT:   %.3844e2.1: type = type_of_inst @F.%.loc10_24.5 [template]
// CHECK:STDOUT:   %.d48a93.1: %.3844e2.1 = splice_inst @F.%.loc10_24.5 [template]
// CHECK:STDOUT:   %.537: type = splice_inst @F.%.loc10_24.8 [template]
// CHECK:STDOUT:   %require_complete.cb0: <witness> = require_complete_type %.537 [template]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %.537 [template]
// CHECK:STDOUT:   %a.patt.970: %pattern_type.fcb = ref_binding_pattern a [template]
// CHECK:STDOUT:   %a.var_patt.98f: %pattern_type.fcb = var_pattern %a.patt.970 [template]
// CHECK:STDOUT:   %.3844e2.2: type = type_of_inst @F.%.loc10_35.3 [template]
// CHECK:STDOUT:   %.d48a93.2: %.3844e2.2 = splice_inst @F.%.loc10_35.3 [template]
// CHECK:STDOUT:   %.9f0: type = type_of_inst @F.%.loc10_36.3 [template]
// CHECK:STDOUT:   %.e5d: %.9f0 = splice_inst @F.%.loc10_36.3 [template]
// CHECK:STDOUT:   %.5d0: type = type_of_inst @F.%.loc10_39.3 [template]
// CHECK:STDOUT:   %.e0a: %.5d0 = splice_inst @F.%.loc10_39.3 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.5d6: %.537 = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.700: type = type_of_inst @F.%.loc10_3.6 [template]
// CHECK:STDOUT:   %.180: %.700 = splice_inst @F.%.loc10_3.6 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ee0935.1: type = splice_block imports.%E.decl [concrete = %E] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.928: type = splice_block %E [concrete = %E] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.9ed: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %a.patt.aa1: %pattern_type.9ed = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt.77c: %pattern_type.9ed = var_pattern %a.patt.aa1 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ee0935.2: type = splice_block imports.%E.decl [concrete = %E] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.E.cpp_overload_set.type: type = cpp_overload_set_type @E.E.cpp_overload_set [concrete]
// CHECK:STDOUT:   %E.E.cpp_overload_set.value: %E.E.cpp_overload_set.type = cpp_overload_set_value @E.E.cpp_overload_set [concrete]
// CHECK:STDOUT:   %inst.name_ref: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %E.ref: %E.E.cpp_overload_set.type = name_ref E, imports.%E.E.cpp_overload_set.value [concrete = %E.E.cpp_overload_set.value]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.db1: type = ptr_type %E [concrete]
// CHECK:STDOUT:   %E__carbon_thunk.type: type = fn_type @E__carbon_thunk [concrete]
// CHECK:STDOUT:   %E__carbon_thunk: %E__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.2c5: init %E = splice_block %.cde {
// CHECK:STDOUT:       %.efc445.2: ref %E = specific_inst @F.%a.var, @F(%i32)
// CHECK:STDOUT:       %addr: %ptr.db1 = addr_of %.efc445.2
// CHECK:STDOUT:       %E__carbon_thunk.call: init %empty_tuple.type = call imports.%E__carbon_thunk.decl(%addr)
// CHECK:STDOUT:       %.cde: init %E to %.efc445.2 = mark_in_place_init %E__carbon_thunk.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.476: init %E = splice_block %.2c5 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block.426619.5) [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.cc5: %SubobjectDestroy.type = facet_value %E, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d63: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cc5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.1cb72a.1: %Destroy.type = facet_value %E, (%DestroyImpl.impl_witness.d63, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7b4: init %empty_tuple.type = splice_block %SelfDestruct.call {
// CHECK:STDOUT:       %.efc445.1: ref %E = specific_inst @F.%a.var, @F(%i32)
// CHECK:STDOUT:       %Destroy.facet.1cb72a.2: %Destroy.type = facet_value %E, (%DestroyImpl.impl_witness.d63, %custom_witness.b80) [concrete = %Destroy.facet.1cb72a.1]
// CHECK:STDOUT:       %.9a9d8c.1: %Destroy.type = converted %E, %Destroy.facet.1cb72a.2 [concrete = %Destroy.facet.1cb72a.1]
// CHECK:STDOUT:       %Destroy.facet.1cb72a.3: %Destroy.type = facet_value %E, (%DestroyImpl.impl_witness.d63, %custom_witness.b80) [concrete = %Destroy.facet.1cb72a.1]
// CHECK:STDOUT:       %.9a9d8c.2: %Destroy.type = converted %E, %Destroy.facet.1cb72a.3 [concrete = %Destroy.facet.1cb72a.1]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.02d4dd.2(%.efc445.1)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .E = %E.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.template: %E.type = struct_value () [concrete = constants.%E.template]
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %E.E.cpp_overload_set.value: %E.E.cpp_overload_set.type = cpp_overload_set_value @E.E.cpp_overload_set [concrete = constants.%E.E.cpp_overload_set.value]
// CHECK:STDOUT:   %E__carbon_thunk.decl: %E__carbon_thunk.type = fn_decl @E__carbon_thunk [concrete = constants.%E__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_15.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_15.2: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc9_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc10_23.2: type = template_inst %T.loc9_15.1 [template = %.loc10_23.2 (constants.%.a92)]
// CHECK:STDOUT:   %.loc10_24.5: <instruction> = call_action (%E.ref.loc10_20, %.loc10_23.1), false [template]
// CHECK:STDOUT:   %.loc10_24.6: type = type_of_inst %.loc10_24.5 [template = %.loc10_24.6 (constants.%.3844e2.1)]
// CHECK:STDOUT:   %.loc10_24.7: @F.%.loc10_24.6 (%.3844e2.1) = splice_inst %.loc10_24.5 [template = %.loc10_24.7 (constants.%.d48a93.1)]
// CHECK:STDOUT:   %.loc10_24.8: <instruction> = convert_to_value_action %.loc10_24.3, type [template]
// CHECK:STDOUT:   %.loc10_24.9: type = splice_inst %.loc10_24.8 [template = %.loc10_24.9 (constants.%.537)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc10_24.9 [template = %require_complete (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc10_24.9 [template = %pattern_type (constants.%pattern_type.fcb)]
// CHECK:STDOUT:   %a.patt.loc10_15.2: @F.%pattern_type (%pattern_type.fcb) = ref_binding_pattern a [template = %a.patt.loc10_15.2 (constants.%a.patt.970)]
// CHECK:STDOUT:   %a.var_patt.loc10_3.2: @F.%pattern_type (%pattern_type.fcb) = var_pattern %a.patt.loc10_15.2 [template = %a.var_patt.loc10_3.2 (constants.%a.var_patt.98f)]
// CHECK:STDOUT:   %.loc10_35.3: <instruction> = call_action (%E.ref.loc10_31, %.loc10_34), false [template]
// CHECK:STDOUT:   %.loc10_35.4: type = type_of_inst %.loc10_35.3 [template = %.loc10_35.4 (constants.%.3844e2.2)]
// CHECK:STDOUT:   %.loc10_35.5: @F.%.loc10_35.4 (%.3844e2.2) = splice_inst %.loc10_35.3 [template = %.loc10_35.5 (constants.%.d48a93.2)]
// CHECK:STDOUT:   %.loc10_36.3: <instruction> = access_member_action %.loc10_35.2, E [template]
// CHECK:STDOUT:   %.loc10_36.4: type = type_of_inst %.loc10_36.3 [template = %.loc10_36.4 (constants.%.9f0)]
// CHECK:STDOUT:   %.loc10_36.5: @F.%.loc10_36.4 (%.9f0) = splice_inst %.loc10_36.3 [template = %.loc10_36.5 (constants.%.e5d)]
// CHECK:STDOUT:   %.loc10_39.3: <instruction> = call_action (%.loc10_36.2), false [template]
// CHECK:STDOUT:   %.loc10_39.4: type = type_of_inst %.loc10_39.3 [template = %.loc10_39.4 (constants.%.5d0)]
// CHECK:STDOUT:   %.loc10_39.5: @F.%.loc10_39.4 (%.5d0) = splice_inst %.loc10_39.3 [template = %.loc10_39.5 (constants.%.e0a)]
// CHECK:STDOUT:   %.loc10_3.4: %tuple.type = initialize_action %.loc10_39.2, %.loc10_24.9, %a.var, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc10_3.4, element0 [template]
// CHECK:STDOUT:   %.loc10_3.5: @F.%.loc10_24.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc10_3.5 (constants.%.5d6)]
// CHECK:STDOUT:   %.loc10_3.6: <instruction> = call_action (constants.%SelfDestruct, %a.var), true [template]
// CHECK:STDOUT:   %.loc10_3.7: type = type_of_inst %.loc10_3.6 [template = %.loc10_3.7 (constants.%.700)]
// CHECK:STDOUT:   %.loc10_3.8: @F.%.loc10_3.7 (%.700) = splice_inst %.loc10_3.6 [template = %.loc10_3.8 (constants.%.180)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.var: ref @F.%.loc10_24.9 (%.537) = var_storage %a.var_patt.loc10_3.1
// CHECK:STDOUT:     %Cpp.ref.loc10_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %E.ref.loc10_31: %E.type = name_ref E, imports.%E.template [concrete = constants.%E.template]
// CHECK:STDOUT:     %T.ref.loc10_34: type = name_ref T, %T.loc9_15.2 [symbolic = %T.loc9_15.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc10_34: type = template_inst %T.ref.loc10_34 [template = %.loc10_23.2 (constants.%.a92)]
// CHECK:STDOUT:     %.loc10_35.1: type = type_of_inst %.loc10_35.3 [template = %.loc10_35.4 (constants.%.3844e2.2)]
// CHECK:STDOUT:     %.loc10_35.2: @F.%.loc10_35.4 (%.3844e2.2) = splice_inst %.loc10_35.3 [template = %.loc10_35.5 (constants.%.d48a93.2)]
// CHECK:STDOUT:     %.loc10_36.1: type = type_of_inst %.loc10_36.3 [template = %.loc10_36.4 (constants.%.9f0)]
// CHECK:STDOUT:     %.loc10_36.2: @F.%.loc10_36.4 (%.9f0) = splice_inst %.loc10_36.3 [template = %.loc10_36.5 (constants.%.e5d)]
// CHECK:STDOUT:     %.loc10_39.1: type = type_of_inst %.loc10_39.3 [template = %.loc10_39.4 (constants.%.5d0)]
// CHECK:STDOUT:     %.loc10_39.2: @F.%.loc10_39.4 (%.5d0) = splice_inst %.loc10_39.3 [template = %.loc10_39.5 (constants.%.e0a)]
// CHECK:STDOUT:     %.loc10_3.1: @F.%.loc10_24.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc10_3.5 (constants.%.5d6)]
// CHECK:STDOUT:     assign %a.var, %.loc10_3.1
// CHECK:STDOUT:     %.loc10_24.1: type = splice_block %.loc10_24.4 [template = %.loc10_24.9 (constants.%.537)] {
// CHECK:STDOUT:       %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %E.ref.loc10_20: %E.type = name_ref E, imports.%E.template [concrete = constants.%E.template]
// CHECK:STDOUT:       %T.ref.loc10_23: type = name_ref T, %T.loc9_15.2 [symbolic = %T.loc9_15.1 (constants.%T.67d)]
// CHECK:STDOUT:       %.loc10_23.1: type = template_inst %T.ref.loc10_23 [template = %.loc10_23.2 (constants.%.a92)]
// CHECK:STDOUT:       %.loc10_24.2: type = type_of_inst %.loc10_24.5 [template = %.loc10_24.6 (constants.%.3844e2.1)]
// CHECK:STDOUT:       %.loc10_24.3: @F.%.loc10_24.6 (%.3844e2.1) = splice_inst %.loc10_24.5 [template = %.loc10_24.7 (constants.%.d48a93.1)]
// CHECK:STDOUT:       %.loc10_24.4: type = splice_inst %.loc10_24.8 [template = %.loc10_24.9 (constants.%.537)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @F.%.loc10_24.9 (%.537) = wrapper_binding a, %a.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc10_15.1: @F.%pattern_type (%pattern_type.fcb) = ref_binding_pattern a [template = %a.patt.loc10_15.2 (constants.%a.patt.970)]
// CHECK:STDOUT:       %a.var_patt.loc10_3.1: @F.%pattern_type (%pattern_type.fcb) = var_pattern %a.patt.loc10_15.1 [template = %a.var_patt.loc10_3.2 (constants.%a.var_patt.98f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10_3.2: type = type_of_inst %.loc10_3.6 [template = %.loc10_3.7 (constants.%.700)]
// CHECK:STDOUT:     %.loc10_3.3: @F.%.loc10_3.7 (%.700) = splice_inst %.loc10_3.6 [template = %.loc10_3.8 (constants.%.180)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %E) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc9_15.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc9_15.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc9_15.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc9_15.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc10_23.2 => constants.%i32
// CHECK:STDOUT:   %.loc10_24.5 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc10_24.6 => type
// CHECK:STDOUT:   %.loc10_24.7 => constants.%E
// CHECK:STDOUT:   %.loc10_24.8 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc10_24.9 => constants.%E
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ed
// CHECK:STDOUT:   %a.patt.loc10_15.2 => constants.%a.patt.aa1
// CHECK:STDOUT:   %a.var_patt.loc10_3.2 => constants.%a.var_patt.77c
// CHECK:STDOUT:   %.loc10_35.3 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc10_35.4 => type
// CHECK:STDOUT:   %.loc10_35.5 => constants.%E
// CHECK:STDOUT:   %.loc10_36.3 => constants.%inst.name_ref
// CHECK:STDOUT:   %.loc10_36.4 => constants.%E.E.cpp_overload_set.type
// CHECK:STDOUT:   %.loc10_36.5 => constants.%E.E.cpp_overload_set.value
// CHECK:STDOUT:   %.loc10_39.3 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc10_39.4 => constants.%E
// CHECK:STDOUT:   %.loc10_39.5 => <not constant>
// CHECK:STDOUT:   %.loc10_3.4 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc10_3.5 => <not constant>
// CHECK:STDOUT:   %.loc10_3.6 => constants.%inst.splice_block.426619.6
// CHECK:STDOUT:   %.loc10_3.7 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc10_3.8 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
