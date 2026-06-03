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

using Aint = A<int>;
using Along = A<long>;

// --- use_class_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_template.h";

fn F(x: Cpp.Aint) -> i32 {
  // Trigger instantiation through name lookup.
  //@dump-sem-ir-begin
  x.f();
  return x.n;
  //@dump-sem-ir-end
}

fn G(p: Cpp.Along*) -> Cpp.Base* {
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

// CHECK:STDOUT: --- use_class_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.971: type = class_type @A.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %A.elem.8ae: type = unbound_element_type %A.971, %i32 [concrete]
// CHECK:STDOUT:   %A.f.cpp_overload_set.type: type = cpp_overload_set_type @A.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %A.f.cpp_overload_set.value: %A.f.cpp_overload_set.type = cpp_overload_set_value @A.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %f__carbon_thunk.type: type = fn_type @f__carbon_thunk [concrete]
// CHECK:STDOUT:   %f__carbon_thunk: %f__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1: %ptr.as.Copy.impl.Op.type.aa4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.14d: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet.14d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %A.a10: type = class_type @A.2 [concrete]
// CHECK:STDOUT:   %ptr.636: type = ptr_type %A.a10 [concrete]
// CHECK:STDOUT:   %ptr.eb6: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.ab1: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f2f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.e51: %ptr.as.Copy.impl.Op.type.f2f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.8ea: %Copy.type = facet_value %ptr.eb6, (%Copy.impl_witness.ab1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.092: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.8ea) [concrete]
// CHECK:STDOUT:   %.470: type = fn_type_with_self_type %Copy.WithSelf.Op.type.092, %Copy.facet.8ea [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.e51, @ptr.as.Copy.impl.Op(%Base) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %A.f.cpp_overload_set.value: %A.f.cpp_overload_set.type = cpp_overload_set_value @A.f.cpp_overload_set [concrete = constants.%A.f.cpp_overload_set.value]
// CHECK:STDOUT:   %f__carbon_thunk.decl: %f__carbon_thunk.type = fn_decl @f__carbon_thunk [concrete = constants.%f__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %A.971) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc9: %A.971 = name_ref x, %x
// CHECK:STDOUT:   %f.ref: %A.f.cpp_overload_set.type = name_ref f, imports.%A.f.cpp_overload_set.value [concrete = constants.%A.f.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %x.ref.loc9, %f.ref
// CHECK:STDOUT:   %f__carbon_thunk.call: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%x.ref.loc9)
// CHECK:STDOUT:   %x.ref.loc10: %A.971 = name_ref x, %x
// CHECK:STDOUT:   %n.ref: %A.elem.8ae = name_ref n, @A.1.%.2 [concrete = @A.1.%.2]
// CHECK:STDOUT:   %.loc10_11.1: ref %i32 = class_element_access %x.ref.loc10, element1
// CHECK:STDOUT:   %.loc10_11.2: %i32 = acquire_value %.loc10_11.1
// CHECK:STDOUT:   %impl.elem0: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.636) -> out %return.param: %ptr.eb6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.636 = name_ref p, %p
// CHECK:STDOUT:   %.loc17_11.1: ref %A.a10 = deref %p.ref
// CHECK:STDOUT:   %.loc17_11.2: ref %Base = class_element_access %.loc17_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.eb6 = addr_of %.loc17_11.2
// CHECK:STDOUT:   %.loc17_11.3: %ptr.eb6 = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.470 = impl_witness_access constants.%Copy.impl_witness.ab1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.e51]
// CHECK:STDOUT:   %bound_method.loc17_11.1: <bound method> = bound_method %.loc17_11.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Base) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_11.2: <bound method> = bound_method %.loc17_11.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.eb6 = call %bound_method.loc17_11.2(%.loc17_11.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_instantiation_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X.548: type = class_type @X.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.62d: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %X.1cb: type = class_type @X.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .XY = %X.decl.138
// CHECK:STDOUT:     .Xint = %X.decl.0ee
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl.138: type = class_decl @X.1 [concrete = constants.%X.548] {} {}
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X.decl.0ee: type = class_decl @X.2 [concrete = constants.%X.1cb] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.104 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.104 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %i32 = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %type.ref.loc7 [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %XY.ref: type = name_ref XY, imports.%X.decl.138 [concrete = constants.%X.548]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %type.ref.loc7: type = name_ref r#type, %i32.4 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %i32 = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.104 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.104 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc21: type = splice_block %type.ref.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Xint.ref: type = name_ref Xint, imports.%X.decl.0ee [concrete = constants.%X.1cb]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %type.ref.loc21: type = name_ref r#type, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %i32 = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc7: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc7: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc7_1.1: <bound method> = bound_method %int_0.loc7, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_1.2: <bound method> = bound_method %int_0.loc7, %specific_fn.loc7 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_1.2(%int_0.loc7) [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   %.loc7: init %i32 = converted %int_0.loc7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   assign file.%x.var, %.loc7
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc21: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc21_1.1: <bound method> = bound_method %int_0.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_1.2: <bound method> = bound_method %int_0.loc21, %specific_fn.loc21 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_1.2(%int_0.loc21) [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   %.loc21: init %i32 = converted %int_0.loc21, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   assign file.%y.var, %.loc21
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
