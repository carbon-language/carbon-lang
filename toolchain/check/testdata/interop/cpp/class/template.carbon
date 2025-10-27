// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/template.carbon

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

// CHECK:STDERR: fail_use_instantiation_error.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./instantiation_error.h:3:25: error: type 'int' cannot be used prior to '::' because it has no members [CppInteropParseError]
// CHECK:STDERR:     3 |   using type = typename T::member;
// CHECK:STDERR:       |                         ^
// CHECK:STDERR: fail_use_instantiation_error.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./instantiation_error.h:2:29: note: in instantiation of template class 'X<int>' requested here [CppInteropParseNote]
// CHECK:STDERR:     2 | template<typename T> struct X {
// CHECK:STDERR:       |                             ^
import Cpp library "instantiation_error.h";

//@dump-sem-ir-begin
var x: Cpp.XY.r#type = 0;

// CHECK:STDERR: fail_use_instantiation_error.carbon:[[@LINE+4]]:8: note: while completing C++ type `Cpp.X` [InCppTypeCompletion]
// CHECK:STDERR: var y: Cpp.Xint.r#type = 0;
// CHECK:STDERR:        ^~~~~~~~~~~~~
// CHECK:STDERR:
var y: Cpp.Xint.r#type = 0;
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_class_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.0bedf0.1: type = class_type @A.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %A.elem.c3f: type = unbound_element_type %A.0bedf0.1, %i32 [concrete]
// CHECK:STDOUT:   %A.f.cpp_overload_set.type: type = cpp_overload_set_type @A.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %A.f.cpp_overload_set.value: %A.f.cpp_overload_set.type = cpp_overload_set_value @A.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const.16f: type = const_type %A.0bedf0.1 [concrete]
// CHECK:STDOUT:   %ptr.703: type = ptr_type %const.16f [concrete]
// CHECK:STDOUT:   %f__carbon_thunk.type: type = fn_type @f__carbon_thunk [concrete]
// CHECK:STDOUT:   %f__carbon_thunk: %f__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.270828.1: type = ptr_type %A.0bedf0.1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c49: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c49 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %A.0bedf0.2: type = class_type @A.2 [concrete]
// CHECK:STDOUT:   %ptr.270828.2: type = ptr_type %A.0bedf0.2 [concrete]
// CHECK:STDOUT:   %ptr.fb2: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.fe1: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Base) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.07b: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.e9f: %Copy.type = facet_value %ptr.fb2, (%Copy.impl_witness.fe1) [concrete]
// CHECK:STDOUT:   %.e43: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.e9f [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.07b, @ptr.as.Copy.impl.Op(%Base) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %A.f.cpp_overload_set.value: %A.f.cpp_overload_set.type = cpp_overload_set_value @A.f.cpp_overload_set [concrete = constants.%A.f.cpp_overload_set.value]
// CHECK:STDOUT:   %f__carbon_thunk.decl: %f__carbon_thunk.type = fn_decl @f__carbon_thunk [concrete = constants.%f__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %A.0bedf0.1) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc9: %A.0bedf0.1 = name_ref x, %x
// CHECK:STDOUT:   %f.ref: %A.f.cpp_overload_set.type = name_ref f, imports.%A.f.cpp_overload_set.value [concrete = constants.%A.f.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %x.ref.loc9, %f.ref
// CHECK:STDOUT:   %.loc9_3: ref %A.0bedf0.1 = value_as_ref %x.ref.loc9
// CHECK:STDOUT:   %addr: %ptr.270828.1 = addr_of %.loc9_3
// CHECK:STDOUT:   %.loc9_7.1: %ptr.703 = as_compatible %addr
// CHECK:STDOUT:   %.loc9_7.2: %ptr.703 = converted %addr, %.loc9_7.1
// CHECK:STDOUT:   %f__carbon_thunk.call: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%.loc9_7.2)
// CHECK:STDOUT:   %x.ref.loc10: %A.0bedf0.1 = name_ref x, %x
// CHECK:STDOUT:   %n.ref: %A.elem.c3f = name_ref n, @A.1.%.2 [concrete = @A.1.%.2]
// CHECK:STDOUT:   %.loc10_11.1: ref %i32 = class_element_access %x.ref.loc10, element1
// CHECK:STDOUT:   %.loc10_11.2: %i32 = bind_value %.loc10_11.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.270828.2) -> %ptr.fb2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.270828.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc17_11.1: ref %A.0bedf0.2 = deref %p.ref
// CHECK:STDOUT:   %.loc17_11.2: ref %Base = class_element_access %.loc17_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.fb2 = addr_of %.loc17_11.2
// CHECK:STDOUT:   %.loc17_11.3: %ptr.fb2 = converted %p.ref, %addr
// CHECK:STDOUT:   %impl.elem0: %.e43 = impl_witness_access constants.%Copy.impl_witness.fe1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.07b]
// CHECK:STDOUT:   %bound_method.loc17_11.1: <bound method> = bound_method %.loc17_11.3, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Base) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_11.2: <bound method> = bound_method %.loc17_11.3, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.fb2 = call %bound_method.loc17_11.2(%.loc17_11.3)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_instantiation_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X.32d59e.1: type = class_type @X.1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %X.32d59e.2: type = class_type @X.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .XY = %X.decl.dec4cb.1
// CHECK:STDOUT:     .Xint = %X.decl.dec4cb.2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl.dec4cb.1: type = class_decl @X.1 [concrete = constants.%X.32d59e.1] {} {}
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X.decl.dec4cb.2: type = class_decl @X.2 [concrete = constants.%X.32d59e.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.7ce = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %i32 = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %type.ref.loc15 [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %XY.ref: type = name_ref XY, imports.%X.decl.dec4cb.1 [concrete = constants.%X.32d59e.1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %type.ref.loc15: type = name_ref r#type, %i32.2 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %i32 = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.7ce = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.7ce = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc21: type = splice_block %type.ref.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Xint.ref: type = name_ref Xint, imports.%X.decl.dec4cb.2 [concrete = constants.%X.32d59e.2]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %type.ref.loc21: type = name_ref r#type, %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %i32 = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc15: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc15: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc15_1.1: <bound method> = bound_method %int_0.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_1.2: <bound method> = bound_method %int_0.loc15, %specific_fn.loc15 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i32 = call %bound_method.loc15_1.2(%int_0.loc15) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc15: init %i32 = converted %int_0.loc15, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign file.%x.var, %.loc15
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc21: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc21_1.1: <bound method> = bound_method %int_0.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_1.2: <bound method> = bound_method %int_0.loc21, %specific_fn.loc21 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_1.2(%int_0.loc21) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc21: init %i32 = converted %int_0.loc21, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign file.%y.var, %.loc21
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
