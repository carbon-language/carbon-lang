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

// CHECK:STDERR: fail_use_instantiation_error.carbon:[[@LINE+12]]:8: error: member access into incomplete class `Cpp.X` [QualifiedExprInIncompleteClassScope]
// CHECK:STDERR: var y: Cpp.Xint.r#type = 0;
// CHECK:STDERR:        ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_use_instantiation_error.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./instantiation_error.h:3:25: note: type 'int' cannot be used prior to '::' because it has no members [CppInteropParseError]
// CHECK:STDERR:     3 |   using type = typename T::member;
// CHECK:STDERR:       |                         ^
// CHECK:STDERR: fail_use_instantiation_error.carbon:[[@LINE-12]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./instantiation_error.h:2:29: note: in instantiation of template class 'X<int>' requested here [CppInteropParseNote]
// CHECK:STDERR:     2 | template<typename T> struct X {
// CHECK:STDERR:       |                             ^
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
struct A {
  T t;
};
''';

//@dump-sem-ir-begin
class B(T: type) {
  var a: Cpp.A(T);
}

fn F(ref b: B(i32)) {
  b.a.t = 123;
}
//@dump-sem-ir-end

// --- let_cpp_template_constructor.carbon

library "[[@TEST_NAME]]";
import Cpp inline '''
template<typename T>
struct A {};
''';

//@dump-sem-ir-begin
fn F(generic T: type) {
  let unused a: Cpp.A(T) = Cpp.A(T).A();
}

fn G() {
  F(i32);
}
//@dump-sem-ir-end

// --- var_cpp_template_constructor.carbon

library "[[@TEST_NAME]]";
import Cpp inline '''
template<typename T>
struct A {};
''';

//@dump-sem-ir-begin
fn F(generic T: type) {
  var unused a: Cpp.A(T) = Cpp.A(T).A();
}
//@dump-sem-ir-end

fn G() {
  F(i32);
}

// CHECK:STDOUT: --- use_class_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.614: type = class_type @A.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %A.elem.708: type = unbound_element_type %A.614, %i32 [concrete]
// CHECK:STDOUT:   %A.f.cpp_overload_set.type: type = cpp_overload_set_type @A.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %A.f.cpp_overload_set.value: %A.f.cpp_overload_set.type = cpp_overload_set_value @A.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %f__carbon_thunk.type: type = fn_type @f__carbon_thunk [concrete]
// CHECK:STDOUT:   %f__carbon_thunk: %f__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.1e1: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.1e1) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet.1e1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %A.e3f: type = class_type @A.2 [concrete]
// CHECK:STDOUT:   %ptr.bc0: type = ptr_type %A.e3f [concrete]
// CHECK:STDOUT:   %ptr.0e0: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.60d: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.136: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.377: %ptr.as.Copy.impl.Op.type.136 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.f1d: %Copy.type = facet_value %ptr.0e0, (%Copy.impl_witness.60d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ff1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.f1d) [concrete]
// CHECK:STDOUT:   %.883: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ff1, %Copy.facet.f1d [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.377, @ptr.as.Copy.impl.Op(%Base) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %A.f.cpp_overload_set.value: %A.f.cpp_overload_set.type = cpp_overload_set_value @A.f.cpp_overload_set [concrete = constants.%A.f.cpp_overload_set.value]
// CHECK:STDOUT:   %f__carbon_thunk.decl: %f__carbon_thunk.type = fn_decl @f__carbon_thunk [concrete = constants.%f__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %A.614) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc9: %A.614 = name_ref x, %x
// CHECK:STDOUT:   %f.ref: %A.f.cpp_overload_set.type = name_ref f, imports.%A.f.cpp_overload_set.value [concrete = constants.%A.f.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %x.ref.loc9, %f.ref
// CHECK:STDOUT:   %f__carbon_thunk.call: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%x.ref.loc9)
// CHECK:STDOUT:   %x.ref.loc10: %A.614 = name_ref x, %x
// CHECK:STDOUT:   %n.ref: %A.elem.708 = name_ref n, @A.1.%field_decl [concrete = @A.1.%field_decl]
// CHECK:STDOUT:   %.loc10_11.1: ref %i32 = class_element_access %x.ref.loc10, element1
// CHECK:STDOUT:   %.loc10_11.2: %i32 = acquire_value %.loc10_11.1
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
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
// CHECK:STDOUT:   %impl.elem0: %.883 = impl_witness_access constants.%Copy.impl_witness.60d, element0 [concrete = constants.%ptr.as.Copy.impl.Op.377]
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
// CHECK:STDOUT:   %X.816: type = class_type @X.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.6b6 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %X.c92: type = class_type @X.2 [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.6b6 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.6b6 = var_pattern %y.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .XY = %X.decl.8c4
// CHECK:STDOUT:     .Xint = %X.decl.b67
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl.8c4: type = class_decl @X.1 [concrete = constants.%X.816] {} {}
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.6b6 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var_storage %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc21: type = splice_block %type.ref.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Xint.ref: type = name_ref Xint, imports.%X.decl.b67 [concrete = constants.%X.c92]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %type.ref.loc21: type = name_ref r#type, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %i32 = wrapper_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.6b6 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.6b6 = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc7: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc7: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc7_1.1: <bound method> = bound_method %int_0.loc7, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_1.2: <bound method> = bound_method %int_0.loc7, %specific_fn.loc7 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_1.2(%int_0.loc7) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc7: init %i32 = converted %int_0.loc7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   assign file.%x.var, %.loc7
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc21: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc21_1.1: <bound method> = bound_method %int_0.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_1.2: <bound method> = bound_method %int_0.loc21, %specific_fn.loc21 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_1.2(%int_0.loc21) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc21: init %i32 = converted %int_0.loc21, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   assign file.%y.var, %.loc21
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_carbon_class_generic_in_cpp_template_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = generic_class_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.60e: type = class_type @B, @B(%T) [symbolic]
// CHECK:STDOUT:   %A.type: type = cpp_type_template_type A [concrete]
// CHECK:STDOUT:   %A.template: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ae6: @B.%.loc12_17.4 (@B.%.loc12_17.4) = splice_inst @B.%.loc12_17.3 [template]
// CHECK:STDOUT:   %.535: type = splice_inst @B.%.loc12_17.6 [template]
// CHECK:STDOUT:   %require_complete.01e: <witness> = require_complete_type %.535 [template]
// CHECK:STDOUT:   %pattern_type.281: type = pattern_type %.535 [template]
// CHECK:STDOUT:   %a.patt.438: %pattern_type.281 = ref_binding_pattern a [template]
// CHECK:STDOUT:   %B.elem.7af: type = unbound_element_type %B.60e, %.535 [template]
// CHECK:STDOUT:   %struct_type.a.be9: type = struct_type {.a: %.535} [template]
// CHECK:STDOUT:   %complete_type.7ab: <witness> = complete_type_witness %struct_type.a.be9 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.bd8: type = class_type @B, @B(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.fdc: type = pattern_type %B.bd8 [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.fdc = ref_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.fdc = wrapper_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %inst.splice_block.72d: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.22e: type = splice_block imports.%A.decl [concrete = %A] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.c26: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.4e4: type = splice_block %A [concrete = %A] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %.ae8: type = custom_layout_type {size=4, align=4, .t@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ece: <witness> = complete_type_witness %.ae8 [concrete]
// CHECK:STDOUT:   %pattern_type.b78: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt.fbf: %pattern_type.b78 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %B.elem.1e0: type = unbound_element_type %B.bd8, %A [concrete]
// CHECK:STDOUT:   %struct_type.a.b63: type = struct_type {.a: %A} [concrete]
// CHECK:STDOUT:   %complete_type.594: <witness> = complete_type_witness %struct_type.a.b63 [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.2bf: %i32 = int_value 123 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.template: %A.type = struct_value () [concrete = constants.%A.template]
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %B.decl: %B.type = class_decl @B [concrete = constants.%B.generic] {
// CHECK:STDOUT:     %T.patt.loc11_10.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: type = splice_block %.loc11_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc11_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.fdc = ref_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.fdc = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: ref %B.bd8 = ref_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %B [concrete = constants.%B.bd8] {
// CHECK:STDOUT:       %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %B: type = class_type @B, @B(constants.%i32) [concrete = constants.%B.bd8]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: ref %B.bd8 = wrapper_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B(%T.loc11_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc11_10.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc11_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc12_17.3: <instruction> = call_template_action clang_decl_id4C000002, (%T.loc11_10.1) [template]
// CHECK:STDOUT:   %.loc12_17.4: type = type_of_inst %.loc12_17.3 [template]
// CHECK:STDOUT:   %.loc12_17.5: @B.%.loc12_17.4 (@B.%.loc12_17.4) = splice_inst %.loc12_17.3 [template = %.loc12_17.5 (constants.%.ae6)]
// CHECK:STDOUT:   %.loc12_17.6: <instruction> = convert_to_value_action %.loc12_17.1, type [template]
// CHECK:STDOUT:   %.loc12_17.7: type = splice_inst %.loc12_17.6 [template = %.loc12_17.7 (constants.%.535)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc12_17.7 [template = %require_complete (constants.%require_complete.01e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc12_17.7 [template = %pattern_type (constants.%pattern_type.281)]
// CHECK:STDOUT:   %a.patt: @B.%pattern_type (%pattern_type.281) = ref_binding_pattern a [template = %a.patt (constants.%a.patt.438)]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T.loc11_10.1) [symbolic = %B (constants.%B.60e)]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %.loc12_17.7 [template = %B.elem (constants.%B.elem.7af)]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: @B.%.loc12_17.7 (%.535)} [template = %struct_type.a (constants.%struct_type.a.be9)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness %struct_type.a [template = %complete_type.loc13_1.2 (constants.%complete_type.7ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %field_decl: @B.%B.elem (%B.elem.7af) = field_decl a, element0, %.loc12_17.2 in [concrete] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, imports.%A.template [concrete = constants.%A.template]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc11_10.2 [symbolic = %T.loc11_10.1 (constants.%T)]
// CHECK:STDOUT:       %.loc12_17.1: @B.%.loc12_17.4 (@B.%.loc12_17.4) = splice_inst %.loc12_17.3 [template = %.loc12_17.5 (constants.%.ae6)]
// CHECK:STDOUT:       %.loc12_17.2: type = splice_inst %.loc12_17.6 [template = %.loc12_17.7 (constants.%.535)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness constants.%struct_type.a.be9 [template = %complete_type.loc13_1.2 (constants.%complete_type.7ab)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B.60e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .a = %field_decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: ref %B.bd8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: ref %B.bd8 = name_ref b, %b
// CHECK:STDOUT:   %a.ref: %B.elem.1e0 = name_ref a, @B.%field_decl [concrete = @B.%field_decl]
// CHECK:STDOUT:   %.loc16_4: ref %A = class_element_access %b.ref, element0
// CHECK:STDOUT:   %t.ref: %A.elem = name_ref t, @A.%field_decl [concrete = @A.%field_decl]
// CHECK:STDOUT:   %.loc16_6: ref %i32 = class_element_access %.loc16_4, element0
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc16_9.1: <bound method> = bound_method %int_123, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_9.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc16_9.2(%int_123) [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc16_9: init %i32 = converted %int_123, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.2bf]
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
// CHECK:STDOUT:   %.loc12_17.3 => constants.%inst.splice_block.72d
// CHECK:STDOUT:   %.loc12_17.4 => type
// CHECK:STDOUT:   %.loc12_17.5 => constants.%A
// CHECK:STDOUT:   %.loc12_17.6 => constants.%inst.splice_block.c26
// CHECK:STDOUT:   %.loc12_17.7 => constants.%A
// CHECK:STDOUT:   %require_complete => constants.%complete_type.ece
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b78
// CHECK:STDOUT:   %a.patt => constants.%a.patt.fbf
// CHECK:STDOUT:   %B => constants.%B.bd8
// CHECK:STDOUT:   %B.elem => constants.%B.elem.1e0
// CHECK:STDOUT:   %struct_type.a => constants.%struct_type.a.b63
// CHECK:STDOUT:   %complete_type.loc13_1.2 => constants.%complete_type.594
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- let_cpp_template_constructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.type: type = cpp_type_template_type A [concrete]
// CHECK:STDOUT:   %A.template: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ae6cd7.1: @F.%.loc10_24.5 (@F.%.loc10_24.5) = splice_inst @F.%.loc10_24.4 [template]
// CHECK:STDOUT:   %.535: type = splice_inst @F.%.loc10_24.7 [template]
// CHECK:STDOUT:   %require_complete.01e: <witness> = require_complete_type %.535 [template]
// CHECK:STDOUT:   %pattern_type.281: type = pattern_type %.535 [template]
// CHECK:STDOUT:   %a.patt.4ab: %pattern_type.281 = value_binding_pattern a [template]
// CHECK:STDOUT:   %.ae6cd7.2: @F.%.loc10_35.3 (@F.%.loc10_35.3) = splice_inst @F.%.loc10_35.2 [template]
// CHECK:STDOUT:   %.4bf: @F.%.loc10_36.3 (@F.%.loc10_36.3) = splice_inst @F.%.loc10_36.2 [template]
// CHECK:STDOUT:   %.7cc: @F.%.loc10_39.4 (@F.%.loc10_39.4) = splice_inst @F.%.loc10_39.3 [template]
// CHECK:STDOUT:   %.14b: %.535 = splice_inst @F.%.loc10_39.6 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%i32) [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %inst.splice_block.72dff3.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.22ed96.1: type = splice_block imports.%A.decl [concrete = %A] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.c26: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.4e4: type = splice_block %A [concrete = %A] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.b78: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt.117: %pattern_type.b78 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %inst.splice_block.72dff3.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.22ed96.2: type = splice_block imports.%A.decl [concrete = %A] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.A.cpp_overload_set.type: type = cpp_overload_set_type @A.A.cpp_overload_set [concrete]
// CHECK:STDOUT:   %A.A.cpp_overload_set.value: %A.A.cpp_overload_set.type = cpp_overload_set_value @A.A.cpp_overload_set [concrete]
// CHECK:STDOUT:   %inst.name_ref: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %A.ref: %A.A.cpp_overload_set.type = name_ref A, imports.%A.A.cpp_overload_set.value [concrete = %A.A.cpp_overload_set.value]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.303: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %A__carbon_thunk.type: type = fn_type @A__carbon_thunk [concrete]
// CHECK:STDOUT:   %A__carbon_thunk: %A__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %inst.splice_block.76c: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.78f: init %A = splice_block %.38d {
// CHECK:STDOUT:       %.3a8: ref %A = temporary_storage
// CHECK:STDOUT:       %addr: %ptr.303 = addr_of %.3a8
// CHECK:STDOUT:       %A__carbon_thunk.call: init %empty_tuple.type = call imports.%A__carbon_thunk.decl(%addr)
// CHECK:STDOUT:       %.38d: init %A to %.3a8 = mark_in_place_init %A__carbon_thunk.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.b90: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.114: %A = splice_block %.f4b {
// CHECK:STDOUT:       %.146: ref %A = temporary %.3a8, %.78f
// CHECK:STDOUT:       %.f4b: %A = acquire_value %.146
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.template: %A.type = struct_value () [concrete = constants.%A.template]
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %A.A.cpp_overload_set.value: %A.A.cpp_overload_set.type = cpp_overload_set_value @A.A.cpp_overload_set [concrete = constants.%A.A.cpp_overload_set.value]
// CHECK:STDOUT:   %A__carbon_thunk.decl: %A__carbon_thunk.type = fn_decl @A__carbon_thunk [concrete = constants.%A__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc9_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc10_24.4: <instruction> = call_template_action clang_decl_id6C000002, (%T.loc9_15.1) [template]
// CHECK:STDOUT:   %.loc10_24.5: type = type_of_inst %.loc10_24.4 [template]
// CHECK:STDOUT:   %.loc10_24.6: @F.%.loc10_24.5 (@F.%.loc10_24.5) = splice_inst %.loc10_24.4 [template = %.loc10_24.6 (constants.%.ae6cd7.1)]
// CHECK:STDOUT:   %.loc10_24.7: <instruction> = convert_to_value_action %.loc10_24.2, type [template]
// CHECK:STDOUT:   %.loc10_24.8: type = splice_inst %.loc10_24.7 [template = %.loc10_24.8 (constants.%.535)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc10_24.8 [template = %require_complete (constants.%require_complete.01e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc10_24.8 [template = %pattern_type (constants.%pattern_type.281)]
// CHECK:STDOUT:   %a.patt.loc10_15.2: @F.%pattern_type (%pattern_type.281) = value_binding_pattern a [template = %a.patt.loc10_15.2 (constants.%a.patt.4ab)]
// CHECK:STDOUT:   %.loc10_35.2: <instruction> = call_template_action clang_decl_id6C000002, (%T.loc9_15.1) [template]
// CHECK:STDOUT:   %.loc10_35.3: type = type_of_inst %.loc10_35.2 [template]
// CHECK:STDOUT:   %.loc10_35.4: @F.%.loc10_35.3 (@F.%.loc10_35.3) = splice_inst %.loc10_35.2 [template = %.loc10_35.4 (constants.%.ae6cd7.2)]
// CHECK:STDOUT:   %.loc10_36.2: <instruction> = access_member_action %.loc10_35.1, A [template]
// CHECK:STDOUT:   %.loc10_36.3: type = type_of_inst %.loc10_36.2 [template]
// CHECK:STDOUT:   %.loc10_36.4: @F.%.loc10_36.3 (@F.%.loc10_36.3) = splice_inst %.loc10_36.2 [template = %.loc10_36.4 (constants.%.4bf)]
// CHECK:STDOUT:   %.loc10_39.3: <instruction> = call_action (%.loc10_36.1), false [template]
// CHECK:STDOUT:   %.loc10_39.4: type = type_of_inst %.loc10_39.3 [template]
// CHECK:STDOUT:   %.loc10_39.5: @F.%.loc10_39.4 (@F.%.loc10_39.4) = splice_inst %.loc10_39.3 [template = %.loc10_39.5 (constants.%.7cc)]
// CHECK:STDOUT:   %.loc10_39.6: <instruction> = convert_to_value_action %.loc10_39.1, %.loc10_24.8 [template]
// CHECK:STDOUT:   %.loc10_39.7: @F.%.loc10_24.8 (%.535) = splice_inst %.loc10_39.6 [template = %.loc10_39.7 (constants.%.14b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Cpp.ref.loc10_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc10_31: %A.type = name_ref A, imports.%A.template [concrete = constants.%A.template]
// CHECK:STDOUT:     %T.ref.loc10_34: type = name_ref T, %T.loc9_15.2 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10_35.1: @F.%.loc10_35.3 (@F.%.loc10_35.3) = splice_inst %.loc10_35.2 [template = %.loc10_35.4 (constants.%.ae6cd7.2)]
// CHECK:STDOUT:     %.loc10_36.1: @F.%.loc10_36.3 (@F.%.loc10_36.3) = splice_inst %.loc10_36.2 [template = %.loc10_36.4 (constants.%.4bf)]
// CHECK:STDOUT:     %.loc10_39.1: @F.%.loc10_39.4 (@F.%.loc10_39.4) = splice_inst %.loc10_39.3 [template = %.loc10_39.5 (constants.%.7cc)]
// CHECK:STDOUT:     %.loc10_39.2: @F.%.loc10_24.8 (%.535) = splice_inst %.loc10_39.6 [template = %.loc10_39.7 (constants.%.14b)]
// CHECK:STDOUT:     %.loc10_24.1: type = splice_block %.loc10_24.3 [template = %.loc10_24.8 (constants.%.535)] {
// CHECK:STDOUT:       %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %A.ref.loc10_20: %A.type = name_ref A, imports.%A.template [concrete = constants.%A.template]
// CHECK:STDOUT:       %T.ref.loc10_23: type = name_ref T, %T.loc9_15.2 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:       %.loc10_24.2: @F.%.loc10_24.5 (@F.%.loc10_24.5) = splice_inst %.loc10_24.4 [template = %.loc10_24.6 (constants.%.ae6cd7.1)]
// CHECK:STDOUT:       %.loc10_24.3: type = splice_inst %.loc10_24.7 [template = %.loc10_24.8 (constants.%.535)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%.loc10_24.8 (%.535) = wrapper_binding a, %.loc10_39.2
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc10_15.1: @F.%pattern_type (%pattern_type.281) = value_binding_pattern a [template = %a.patt.loc10_15.2 (constants.%a.patt.4ab)]
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
// CHECK:STDOUT:   %.loc10_24.4 => constants.%inst.splice_block.72dff3.1
// CHECK:STDOUT:   %.loc10_24.5 => type
// CHECK:STDOUT:   %.loc10_24.6 => constants.%A
// CHECK:STDOUT:   %.loc10_24.7 => constants.%inst.splice_block.c26
// CHECK:STDOUT:   %.loc10_24.8 => constants.%A
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b78
// CHECK:STDOUT:   %a.patt.loc10_15.2 => constants.%a.patt.117
// CHECK:STDOUT:   %.loc10_35.2 => constants.%inst.splice_block.72dff3.2
// CHECK:STDOUT:   %.loc10_35.3 => type
// CHECK:STDOUT:   %.loc10_35.4 => constants.%A
// CHECK:STDOUT:   %.loc10_36.2 => constants.%inst.name_ref
// CHECK:STDOUT:   %.loc10_36.3 => constants.%A.A.cpp_overload_set.type
// CHECK:STDOUT:   %.loc10_36.4 => constants.%A.A.cpp_overload_set.value
// CHECK:STDOUT:   %.loc10_39.3 => constants.%inst.splice_block.76c
// CHECK:STDOUT:   %.loc10_39.4 => constants.%A
// CHECK:STDOUT:   %.loc10_39.5 => invalid
// CHECK:STDOUT:   %.loc10_39.6 => constants.%inst.splice_block.b90
// CHECK:STDOUT:   %.loc10_39.7 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_cpp_template_constructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.type: type = cpp_type_template_type A [concrete]
// CHECK:STDOUT:   %A.template: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ae6cd7.1: @F.%.loc10_24.5 (@F.%.loc10_24.5) = splice_inst @F.%.loc10_24.4 [template]
// CHECK:STDOUT:   %.535: type = splice_inst @F.%.loc10_24.7 [template]
// CHECK:STDOUT:   %require_complete.01e: <witness> = require_complete_type %.535 [template]
// CHECK:STDOUT:   %pattern_type.281: type = pattern_type %.535 [template]
// CHECK:STDOUT:   %a.patt.44a: %pattern_type.281 = ref_binding_pattern a [template]
// CHECK:STDOUT:   %a.var_patt.5fb: %pattern_type.281 = var_pattern %a.patt.44a [template]
// CHECK:STDOUT:   %.ae6cd7.2: @F.%.loc10_35.3 (@F.%.loc10_35.3) = splice_inst @F.%.loc10_35.2 [template]
// CHECK:STDOUT:   %.4bf: @F.%.loc10_36.3 (@F.%.loc10_36.3) = splice_inst @F.%.loc10_36.2 [template]
// CHECK:STDOUT:   %.7cc: @F.%.loc10_39.3 (@F.%.loc10_39.3) = splice_inst @F.%.loc10_39.2 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.847f2c.1: @F.%.loc10_3.12 (@F.%.loc10_3.12) = splice_inst @F.%.loc10_3.11 [template]
// CHECK:STDOUT:   %.7fa0c8.1: @F.%.loc10_3.15 (@F.%.loc10_3.15) = splice_inst @F.%.loc10_3.14 [template]
// CHECK:STDOUT:   %.8da: @F.%.loc10_3.18 (@F.%.loc10_3.18) = splice_inst @F.%.loc10_3.17 [template]
// CHECK:STDOUT:   %.9e9: @F.%.loc10_3.21 (@F.%.loc10_3.21) = splice_inst @F.%.loc10_3.20 [template]
// CHECK:STDOUT:   %.9f8: %.535 = splice_inst @F.%.loc10_3.23 [template]
// CHECK:STDOUT:   %.073: %.535 = splice_inst @F.%.loc10_3.25 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.cb8: %.535 = splice_inst @F.%.loc10_3.27 [template]
// CHECK:STDOUT:   %.495: @F.%.loc10_3.30 (@F.%.loc10_3.30) = splice_inst @F.%.loc10_3.29 [template]
// CHECK:STDOUT:   %.013: @F.%.loc10_3.33 (@F.%.loc10_3.33) = splice_inst @F.%.loc10_3.32 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %inst.splice_block.72dff3.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.22ed96.1: type = splice_block imports.%A.decl [concrete = %A] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.c26: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.4e4: type = splice_block %A [concrete = %A] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.b78: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt.198: %pattern_type.b78 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt.47b: %pattern_type.b78 = var_pattern %a.patt.198 [concrete]
// CHECK:STDOUT:   %inst.splice_block.72dff3.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.22ed96.2: type = splice_block imports.%A.decl [concrete = %A] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.A.cpp_overload_set.type: type = cpp_overload_set_type @A.A.cpp_overload_set [concrete]
// CHECK:STDOUT:   %A.A.cpp_overload_set.value: %A.A.cpp_overload_set.type = cpp_overload_set_value @A.A.cpp_overload_set [concrete]
// CHECK:STDOUT:   %inst.name_ref: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %A.ref: %A.A.cpp_overload_set.type = name_ref A, imports.%A.A.cpp_overload_set.value [concrete = %A.A.cpp_overload_set.value]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.303: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %A__carbon_thunk.type: type = fn_type @A__carbon_thunk [concrete]
// CHECK:STDOUT:   %A__carbon_thunk: %A__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %inst.splice_block.76c: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.78f: init %A = splice_block %.38d {
// CHECK:STDOUT:       %.3a8: ref %A = temporary_storage
// CHECK:STDOUT:       %addr: %ptr.303 = addr_of %.3a8
// CHECK:STDOUT:       %A__carbon_thunk.call: init %empty_tuple.type = call imports.%A__carbon_thunk.decl(%addr)
// CHECK:STDOUT:       %.38d: init %A to %.3a8 = mark_in_place_init %A__carbon_thunk.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.04f: <instruction> = call_action (%ImplicitAs.generic, %.535), false [template]
// CHECK:STDOUT:   %.5a1: type = type_of_inst %.04f [template]
// CHECK:STDOUT:   %.847f2c.2: %.5a1 = splice_inst %.04f [template]
// CHECK:STDOUT:   %.5d9: <instruction> = access_member_action %.847f2c.2, Convert [template]
// CHECK:STDOUT:   %.831: type = type_of_inst %.5d9 [template]
// CHECK:STDOUT:   %.7fa0c8.2: %.831 = splice_inst %.5d9 [template]
// CHECK:STDOUT:   %.98e: <instruction> = compound_member_access_action %.78f, %.7fa0c8.2 [template]
// CHECK:STDOUT:   %.81d: type = type_of_inst %.98e [template]
// CHECK:STDOUT:   %.472: %.81d = splice_inst %.98e [template]
// CHECK:STDOUT:   %.6bc: <instruction> = call_action (%.472), true [template]
// CHECK:STDOUT:   %.133: type = type_of_inst %.6bc [template]
// CHECK:STDOUT:   %.da3: %.133 = splice_inst %.6bc [template]
// CHECK:STDOUT:   %.d73: %A = splice_inst %.6bc [template]
// CHECK:STDOUT:   %inst.as_compatible.7b3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.4c2: %A = as_compatible %.da3 [template = %.d73]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.33e: <instruction> = convert_to_category_action %.d73, element10 [template]
// CHECK:STDOUT:   %.4ee: %A = splice_inst %.33e [template]
// CHECK:STDOUT:   %inst.as_compatible.b4e: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.463: ref %A = as_compatible @F.%a.var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.cpp_destructor.type: type = fn_type @A.cpp_destructor [concrete]
// CHECK:STDOUT:   %A.cpp_destructor: %A.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.294: <witness> = custom_witness (%A.cpp_destructor), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.bf5: %Destroy.type = facet_value %A, (%custom_witness.294) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d24: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.bf5) [concrete]
// CHECK:STDOUT:   %.429: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d24, %Destroy.facet.bf5 [concrete]
// CHECK:STDOUT:   %inst.splice_block.688: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.471: <bound method> = splice_block %bound_method {
// CHECK:STDOUT:       %impl.elem0: %.429 = impl_witness_access %custom_witness.294, element0 [concrete = %A.cpp_destructor]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.463, %impl.elem0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %A.cpp_destructor.call: init %empty_tuple.type = call %.471(%.463)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.template: %A.type = struct_value () [concrete = constants.%A.template]
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %A.A.cpp_overload_set.value: %A.A.cpp_overload_set.type = cpp_overload_set_value @A.A.cpp_overload_set [concrete = constants.%A.A.cpp_overload_set.value]
// CHECK:STDOUT:   %A__carbon_thunk.decl: %A__carbon_thunk.type = fn_decl @A__carbon_thunk [concrete = constants.%A__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc9_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc10_24.4: <instruction> = call_template_action clang_decl_id5C000002, (%T.loc9_15.1) [template]
// CHECK:STDOUT:   %.loc10_24.5: type = type_of_inst %.loc10_24.4 [template]
// CHECK:STDOUT:   %.loc10_24.6: @F.%.loc10_24.5 (@F.%.loc10_24.5) = splice_inst %.loc10_24.4 [template = %.loc10_24.6 (constants.%.ae6cd7.1)]
// CHECK:STDOUT:   %.loc10_24.7: <instruction> = convert_to_value_action %.loc10_24.2, type [template]
// CHECK:STDOUT:   %.loc10_24.8: type = splice_inst %.loc10_24.7 [template = %.loc10_24.8 (constants.%.535)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc10_24.8 [template = %require_complete (constants.%require_complete.01e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc10_24.8 [template = %pattern_type (constants.%pattern_type.281)]
// CHECK:STDOUT:   %a.patt.loc10_15.2: @F.%pattern_type (%pattern_type.281) = ref_binding_pattern a [template = %a.patt.loc10_15.2 (constants.%a.patt.44a)]
// CHECK:STDOUT:   %a.var_patt.loc10_3.2: @F.%pattern_type (%pattern_type.281) = var_pattern %a.patt.loc10_15.2 [template = %a.var_patt.loc10_3.2 (constants.%a.var_patt.5fb)]
// CHECK:STDOUT:   %.loc10_35.2: <instruction> = call_template_action clang_decl_id5C000002, (%T.loc9_15.1) [template]
// CHECK:STDOUT:   %.loc10_35.3: type = type_of_inst %.loc10_35.2 [template]
// CHECK:STDOUT:   %.loc10_35.4: @F.%.loc10_35.3 (@F.%.loc10_35.3) = splice_inst %.loc10_35.2 [template = %.loc10_35.4 (constants.%.ae6cd7.2)]
// CHECK:STDOUT:   %.loc10_36.2: <instruction> = access_member_action %.loc10_35.1, A [template]
// CHECK:STDOUT:   %.loc10_36.3: type = type_of_inst %.loc10_36.2 [template]
// CHECK:STDOUT:   %.loc10_36.4: @F.%.loc10_36.3 (@F.%.loc10_36.3) = splice_inst %.loc10_36.2 [template = %.loc10_36.4 (constants.%.4bf)]
// CHECK:STDOUT:   %.loc10_39.2: <instruction> = call_action (%.loc10_36.1), false [template]
// CHECK:STDOUT:   %.loc10_39.3: type = type_of_inst %.loc10_39.2 [template]
// CHECK:STDOUT:   %.loc10_39.4: @F.%.loc10_39.3 (@F.%.loc10_39.3) = splice_inst %.loc10_39.2 [template = %.loc10_39.4 (constants.%.7cc)]
// CHECK:STDOUT:   %.loc10_3.11: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.535), false [template]
// CHECK:STDOUT:   %.loc10_3.12: type = type_of_inst %.loc10_3.11 [template]
// CHECK:STDOUT:   %.loc10_3.13: @F.%.loc10_3.12 (@F.%.loc10_3.12) = splice_inst %.loc10_3.11 [template = %.loc10_3.13 (constants.%.847f2c.1)]
// CHECK:STDOUT:   %.loc10_3.14: <instruction> = access_member_action %.loc10_3.1, Convert [template]
// CHECK:STDOUT:   %.loc10_3.15: type = type_of_inst %.loc10_3.14 [template]
// CHECK:STDOUT:   %.loc10_3.16: @F.%.loc10_3.15 (@F.%.loc10_3.15) = splice_inst %.loc10_3.14 [template = %.loc10_3.16 (constants.%.7fa0c8.1)]
// CHECK:STDOUT:   %.loc10_3.17: <instruction> = compound_member_access_action %.loc10_39.1, %.loc10_3.2 [template]
// CHECK:STDOUT:   %.loc10_3.18: type = type_of_inst %.loc10_3.17 [template]
// CHECK:STDOUT:   %.loc10_3.19: @F.%.loc10_3.18 (@F.%.loc10_3.18) = splice_inst %.loc10_3.17 [template = %.loc10_3.19 (constants.%.8da)]
// CHECK:STDOUT:   %.loc10_3.20: <instruction> = call_action (%.loc10_3.3), true [template]
// CHECK:STDOUT:   %.loc10_3.21: type = type_of_inst %.loc10_3.20 [template]
// CHECK:STDOUT:   %.loc10_3.22: @F.%.loc10_3.21 (@F.%.loc10_3.21) = splice_inst %.loc10_3.20 [template = %.loc10_3.22 (constants.%.9e9)]
// CHECK:STDOUT:   %.loc10_3.23: <instruction> = refine_type_action %.loc10_3.5, %.loc10_24.8 [template]
// CHECK:STDOUT:   %.loc10_3.24: @F.%.loc10_24.8 (%.535) = splice_inst %.loc10_3.23 [template = %.loc10_3.24 (constants.%.9f8)]
// CHECK:STDOUT:   %.loc10_3.25: <instruction> = convert_to_category_action %.loc10_3.6, element10 [template]
// CHECK:STDOUT:   %.loc10_3.26: @F.%.loc10_24.8 (%.535) = splice_inst %.loc10_3.25 [template = %.loc10_3.26 (constants.%.073)]
// CHECK:STDOUT:   %.loc10_3.27: <instruction> = refine_type_action %a.var, %.loc10_24.8 [template]
// CHECK:STDOUT:   %.loc10_3.28: @F.%.loc10_24.8 (%.535) = splice_inst %.loc10_3.27 [template = %.loc10_3.28 (constants.%.cb8)]
// CHECK:STDOUT:   %.loc10_3.29: <instruction> = compound_member_access_action %.loc10_3.8, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc10_3.30: type = type_of_inst %.loc10_3.29 [template]
// CHECK:STDOUT:   %.loc10_3.31: @F.%.loc10_3.30 (@F.%.loc10_3.30) = splice_inst %.loc10_3.29 [template = %.loc10_3.31 (constants.%.495)]
// CHECK:STDOUT:   %.loc10_3.32: <instruction> = call_action (%.loc10_3.9), true [template]
// CHECK:STDOUT:   %.loc10_3.33: type = type_of_inst %.loc10_3.32 [template]
// CHECK:STDOUT:   %.loc10_3.34: @F.%.loc10_3.33 (@F.%.loc10_3.33) = splice_inst %.loc10_3.32 [template = %.loc10_3.34 (constants.%.013)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.var: ref @F.%.loc10_24.8 (%.535) = var_storage %a.var_patt.loc10_3.1
// CHECK:STDOUT:     %Cpp.ref.loc10_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc10_31: %A.type = name_ref A, imports.%A.template [concrete = constants.%A.template]
// CHECK:STDOUT:     %T.ref.loc10_34: type = name_ref T, %T.loc9_15.2 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10_35.1: @F.%.loc10_35.3 (@F.%.loc10_35.3) = splice_inst %.loc10_35.2 [template = %.loc10_35.4 (constants.%.ae6cd7.2)]
// CHECK:STDOUT:     %.loc10_36.1: @F.%.loc10_36.3 (@F.%.loc10_36.3) = splice_inst %.loc10_36.2 [template = %.loc10_36.4 (constants.%.4bf)]
// CHECK:STDOUT:     %.loc10_39.1: @F.%.loc10_39.3 (@F.%.loc10_39.3) = splice_inst %.loc10_39.2 [template = %.loc10_39.4 (constants.%.7cc)]
// CHECK:STDOUT:     %.loc10_3.1: @F.%.loc10_3.12 (@F.%.loc10_3.12) = splice_inst %.loc10_3.11 [template = %.loc10_3.13 (constants.%.847f2c.1)]
// CHECK:STDOUT:     %.loc10_3.2: @F.%.loc10_3.15 (@F.%.loc10_3.15) = splice_inst %.loc10_3.14 [template = %.loc10_3.16 (constants.%.7fa0c8.1)]
// CHECK:STDOUT:     %.loc10_3.3: @F.%.loc10_3.18 (@F.%.loc10_3.18) = splice_inst %.loc10_3.17 [template = %.loc10_3.19 (constants.%.8da)]
// CHECK:STDOUT:     %.loc10_3.4: @F.%.loc10_3.21 (@F.%.loc10_3.21) = splice_inst %.loc10_3.20 [template = %.loc10_3.22 (constants.%.9e9)]
// CHECK:STDOUT:     %.loc10_3.5: @F.%.loc10_24.8 (%.535) = converted %.loc10_39.1, %.loc10_3.4 [template = %.loc10_3.22 (constants.%.9e9)]
// CHECK:STDOUT:     %.loc10_3.6: @F.%.loc10_24.8 (%.535) = splice_inst %.loc10_3.23 [template = %.loc10_3.24 (constants.%.9f8)]
// CHECK:STDOUT:     %.loc10_3.7: @F.%.loc10_24.8 (%.535) = splice_inst %.loc10_3.25 [template = %.loc10_3.26 (constants.%.073)]
// CHECK:STDOUT:     assign %a.var, %.loc10_3.7
// CHECK:STDOUT:     %.loc10_24.1: type = splice_block %.loc10_24.3 [template = %.loc10_24.8 (constants.%.535)] {
// CHECK:STDOUT:       %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %A.ref.loc10_20: %A.type = name_ref A, imports.%A.template [concrete = constants.%A.template]
// CHECK:STDOUT:       %T.ref.loc10_23: type = name_ref T, %T.loc9_15.2 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:       %.loc10_24.2: @F.%.loc10_24.5 (@F.%.loc10_24.5) = splice_inst %.loc10_24.4 [template = %.loc10_24.6 (constants.%.ae6cd7.1)]
// CHECK:STDOUT:       %.loc10_24.3: type = splice_inst %.loc10_24.7 [template = %.loc10_24.8 (constants.%.535)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @F.%.loc10_24.8 (%.535) = wrapper_binding a, %a.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc10_15.1: @F.%pattern_type (%pattern_type.281) = ref_binding_pattern a [template = %a.patt.loc10_15.2 (constants.%a.patt.44a)]
// CHECK:STDOUT:       %a.var_patt.loc10_3.1: @F.%pattern_type (%pattern_type.281) = var_pattern %a.patt.loc10_15.1 [template = %a.var_patt.loc10_3.2 (constants.%a.var_patt.5fb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10_3.8: @F.%.loc10_24.8 (%.535) = splice_inst %.loc10_3.27 [template = %.loc10_3.28 (constants.%.cb8)]
// CHECK:STDOUT:     %.loc10_3.9: @F.%.loc10_3.30 (@F.%.loc10_3.30) = splice_inst %.loc10_3.29 [template = %.loc10_3.31 (constants.%.495)]
// CHECK:STDOUT:     %.loc10_3.10: @F.%.loc10_3.33 (@F.%.loc10_3.33) = splice_inst %.loc10_3.32 [template = %.loc10_3.34 (constants.%.013)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.loc10_24.4 => constants.%inst.splice_block.72dff3.1
// CHECK:STDOUT:   %.loc10_24.5 => type
// CHECK:STDOUT:   %.loc10_24.6 => constants.%A
// CHECK:STDOUT:   %.loc10_24.7 => constants.%inst.splice_block.c26
// CHECK:STDOUT:   %.loc10_24.8 => constants.%A
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b78
// CHECK:STDOUT:   %a.patt.loc10_15.2 => constants.%a.patt.198
// CHECK:STDOUT:   %a.var_patt.loc10_3.2 => constants.%a.var_patt.47b
// CHECK:STDOUT:   %.loc10_35.2 => constants.%inst.splice_block.72dff3.2
// CHECK:STDOUT:   %.loc10_35.3 => type
// CHECK:STDOUT:   %.loc10_35.4 => constants.%A
// CHECK:STDOUT:   %.loc10_36.2 => constants.%inst.name_ref
// CHECK:STDOUT:   %.loc10_36.3 => constants.%A.A.cpp_overload_set.type
// CHECK:STDOUT:   %.loc10_36.4 => constants.%A.A.cpp_overload_set.value
// CHECK:STDOUT:   %.loc10_39.2 => constants.%inst.splice_block.76c
// CHECK:STDOUT:   %.loc10_39.3 => constants.%A
// CHECK:STDOUT:   %.loc10_39.4 => invalid
// CHECK:STDOUT:   %.loc10_3.11 => constants.%.04f
// CHECK:STDOUT:   %.loc10_3.12 => constants.%.5a1
// CHECK:STDOUT:   %.loc10_3.13 => constants.%.847f2c.2
// CHECK:STDOUT:   %.loc10_3.14 => constants.%.5d9
// CHECK:STDOUT:   %.loc10_3.15 => constants.%.831
// CHECK:STDOUT:   %.loc10_3.16 => constants.%.7fa0c8.2
// CHECK:STDOUT:   %.loc10_3.17 => constants.%.98e
// CHECK:STDOUT:   %.loc10_3.18 => constants.%.81d
// CHECK:STDOUT:   %.loc10_3.19 => constants.%.472
// CHECK:STDOUT:   %.loc10_3.20 => constants.%.6bc
// CHECK:STDOUT:   %.loc10_3.21 => constants.%.133
// CHECK:STDOUT:   %.loc10_3.22 => constants.%.da3
// CHECK:STDOUT:   %.loc10_3.23 => constants.%inst.as_compatible.7b3
// CHECK:STDOUT:   %.loc10_3.24 => constants.%.d73
// CHECK:STDOUT:   %.loc10_3.25 => constants.%.33e
// CHECK:STDOUT:   %.loc10_3.26 => constants.%.4ee
// CHECK:STDOUT:   %.loc10_3.27 => constants.%inst.as_compatible.b4e
// CHECK:STDOUT:   %.loc10_3.28 => invalid
// CHECK:STDOUT:   %.loc10_3.29 => constants.%inst.splice_block.688
// CHECK:STDOUT:   %.loc10_3.30 => <bound method>
// CHECK:STDOUT:   %.loc10_3.31 => invalid
// CHECK:STDOUT:   %.loc10_3.32 => constants.%inst.call
// CHECK:STDOUT:   %.loc10_3.33 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc10_3.34 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
