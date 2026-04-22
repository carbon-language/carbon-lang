// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/import/indirect.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/import/indirect.carbon

// --- shared.h

namespace A {
  struct X {
    X(int);
    int g();
    int x;
  };

  template<typename T>
  struct Y {
    T y;
  };

  struct Z {
    int z;
  };
}

// --- direct_import.carbon

library "[[@TEST_NAME]]";

import Cpp library "shared.h";

fn F() -> Cpp.A.X { return 1; }

fn G() -> Cpp.A.Y(i32);

fn H() -> Cpp.A.Y(Cpp.A.Z);

alias AX = Cpp.A.X;

// --- indirect_import_via_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "shared.h";
import library "direct_import";

fn Field() -> i32 {
  //@dump-sem-ir-begin
  return F().x;
  //@dump-sem-ir-end
}

fn Function() -> i32 {
  //@dump-sem-ir-begin
  return F().g();
  //@dump-sem-ir-end
}

// --- fail_todo_not_included.carbon

library "[[@TEST_NAME]]";

import Cpp;
// CHECK:STDERR: fail_todo_not_included.carbon:[[@LINE+6]]:1: in import [InImport]
// CHECK:STDERR: direct_import.carbon:4:10: in file included here [InCppInclude]
// CHECK:STDERR: ./shared.h:3:10: error: semantics TODO: `use of imported C++ declaration with no corresponding local import` [SemanticsTodo]
// CHECK:STDERR:   struct X {
// CHECK:STDERR:          ^
// CHECK:STDERR:
import library "direct_import";

fn Field() -> i32 {
  // TODO: This should eventually work, by importing the C++ AST from
  // `direct_import`.
  return F().x;
}

// --- fail_todo_no_cpp_import.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_no_cpp_import.carbon:[[@LINE+6]]:1: in import [InImport]
// CHECK:STDERR: direct_import.carbon:4:10: in file included here [InCppInclude]
// CHECK:STDERR: ./shared.h:3:10: error: semantics TODO: `indirect import of C++ declaration with no direct Cpp import` [SemanticsTodo]
// CHECK:STDERR:   struct X {
// CHECK:STDERR:          ^
// CHECK:STDERR:
import library "direct_import";

fn Field() -> i32 {
  // TODO: This should eventually work, by importing the C++ AST from
  // `direct_import`, even though we don't otherwise have any C++ imports in
  // this compilation.
  return F().x;
}

// --- indirect_import_via_alias.carbon

library "[[@TEST_NAME]]";

import Cpp library "shared.h";
import library "direct_import";

fn Use() -> (i32, i32) {
  //@dump-sem-ir-begin
  var ax: AX = 1;
  return (ax.x, ax.g());
  //@dump-sem-ir-end
}

// --- import_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "shared.h";
import library "direct_import";

fn Use() -> i32 {
  //@dump-sem-ir-begin
  return G().y;
  //@dump-sem-ir-end
}

fn UseZ() -> Cpp.A.Z {
  //@dump-sem-ir-begin
  return H().y;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- indirect_import_via_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %pattern_type.f80: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %X.cpp_destructor.type: type = fn_type @X.cpp_destructor [concrete]
// CHECK:STDOUT:   %X.cpp_destructor: %X.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.type: type = cpp_overload_set_type @X.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.value: %X.g.cpp_overload_set.type = cpp_overload_set_value @X.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.g.type: type = fn_type @X.g [concrete]
// CHECK:STDOUT:   %X.g: %X.g.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//direct_import, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.value: %X.g.cpp_overload_set.type = cpp_overload_set_value @X.g.cpp_overload_set [concrete = constants.%X.g.cpp_overload_set.value]
// CHECK:STDOUT:   %X.g.decl: %X.g.type = fn_decl @X.g [concrete = constants.%X.g] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f80 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.f80 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Field() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_12.1: ref %X = temporary_storage
// CHECK:STDOUT:   %F.call: init %X to %.loc9_12.1 = call %F.ref()
// CHECK:STDOUT:   %.loc9_12.2: ref %X = temporary %.loc9_12.1, %F.call
// CHECK:STDOUT:   %x.ref: %X.elem = name_ref x, @X.%.1 [concrete = @X.%.1]
// CHECK:STDOUT:   %.loc9_13.1: ref %i32 = class_element_access %.loc9_12.2, element0
// CHECK:STDOUT:   %.loc9_13.2: %i32 = acquire_value %.loc9_13.1
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc9_13.1: <bound method> = bound_method %.loc9_13.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_13.2: <bound method> = bound_method %.loc9_13.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc9_13.2(%.loc9_13.2)
// CHECK:STDOUT:   %X.cpp_destructor.bound: <bound method> = bound_method %.loc9_12.2, constants.%X.cpp_destructor
// CHECK:STDOUT:   %X.cpp_destructor.call: init %empty_tuple.type = call %X.cpp_destructor.bound(%.loc9_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc15_12.1: ref %X = temporary_storage
// CHECK:STDOUT:   %F.call: init %X to %.loc15_12.1 = call %F.ref()
// CHECK:STDOUT:   %.loc15_12.2: ref %X = temporary %.loc15_12.1, %F.call
// CHECK:STDOUT:   %g.ref: %X.g.cpp_overload_set.type = name_ref g, imports.%X.g.cpp_overload_set.value [concrete = constants.%X.g.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc15_12.2, %g.ref
// CHECK:STDOUT:   %X.g.call: init %i32 = call imports.%X.g.decl(%.loc15_12.2)
// CHECK:STDOUT:   %X.cpp_destructor.bound: <bound method> = bound_method %.loc15_12.2, constants.%X.cpp_destructor
// CHECK:STDOUT:   %X.cpp_destructor.call: init %empty_tuple.type = call %X.cpp_destructor.bound(%.loc15_12.2)
// CHECK:STDOUT:   return %X.g.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_import_via_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.f80: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.611: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %X__carbon_thunk.type: type = fn_type @X__carbon_thunk [concrete]
// CHECK:STDOUT:   %X__carbon_thunk: %X__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.type: type = cpp_overload_set_type @X.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.value: %X.g.cpp_overload_set.type = cpp_overload_set_value @X.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.g.type: type = fn_type @X.g [concrete]
// CHECK:STDOUT:   %X.g: %X.g.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %X.cpp_destructor.type: type = fn_type @X.cpp_destructor [concrete]
// CHECK:STDOUT:   %X.cpp_destructor: %X.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.AX: type = import_ref Main//direct_import, AX, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %X__carbon_thunk.decl: %X__carbon_thunk.type = fn_decl @X__carbon_thunk [concrete = constants.%X__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.value: %X.g.cpp_overload_set.type = cpp_overload_set_value @X.g.cpp_overload_set [concrete = constants.%X.g.cpp_overload_set.value]
// CHECK:STDOUT:   %X.g.decl: %X.g.type = fn_decl @X.g [concrete = constants.%X.g] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f80 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.f80 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() -> out %return.param: %tuple.type.d07 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ax.patt: %pattern_type.f80 = ref_binding_pattern ax [concrete]
// CHECK:STDOUT:     %ax.var_patt: %pattern_type.f80 = var_pattern %ax.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ax.var: ref %X = var %ax.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_3.1: ref %X = splice_block %ax.var {}
// CHECK:STDOUT:   %impl.elem0.loc9: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_16.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %int_1, %.loc9_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %addr: %ptr.611 = addr_of %.loc9_3.1
// CHECK:STDOUT:   %X__carbon_thunk.call: init %empty_tuple.type = call imports.%X__carbon_thunk.decl(%.loc9_16.2, %addr)
// CHECK:STDOUT:   %.loc9_3.2: init %X to %.loc9_3.1 = mark_in_place_init %X__carbon_thunk.call
// CHECK:STDOUT:   %.loc9_3.3: init %X = converted %int_1, %.loc9_3.2
// CHECK:STDOUT:   assign %ax.var, %.loc9_3.3
// CHECK:STDOUT:   %AX.ref: type = name_ref AX, imports.%Main.AX [concrete = constants.%X]
// CHECK:STDOUT:   %ax: ref %X = ref_binding ax, %ax.var
// CHECK:STDOUT:   %ax.ref.loc10_11: ref %X = name_ref ax, %ax
// CHECK:STDOUT:   %x.ref: %X.elem = name_ref x, @X.%.1 [concrete = @X.%.1]
// CHECK:STDOUT:   %.loc10_13.1: ref %i32 = class_element_access %ax.ref.loc10_11, element0
// CHECK:STDOUT:   %ax.ref.loc10_17: ref %X = name_ref ax, %ax
// CHECK:STDOUT:   %g.ref: %X.g.cpp_overload_set.type = name_ref g, imports.%X.g.cpp_overload_set.value [concrete = constants.%X.g.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10_19: <bound method> = bound_method %ax.ref.loc10_17, %g.ref
// CHECK:STDOUT:   %X.g.call: init %i32 = call imports.%X.g.decl(%ax.ref.loc10_17)
// CHECK:STDOUT:   %.loc10_23.1: %tuple.type.d07 = tuple_literal (%.loc10_13.1, %X.g.call)
// CHECK:STDOUT:   %.loc10_13.2: %i32 = acquire_value %.loc10_13.1
// CHECK:STDOUT:   %impl.elem0.loc10: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc10_13.1: <bound method> = bound_method %.loc10_13.2, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_13.2: <bound method> = bound_method %.loc10_13.2, %specific_fn.loc10
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_13.2(%.loc10_13.2)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc10_23.2: init %i32 to %tuple.elem0 = in_place_init %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc10_23.3: init %i32 to %tuple.elem1 = in_place_init %X.g.call
// CHECK:STDOUT:   %.loc10_23.4: init %tuple.type.d07 to %return.param = tuple_init (%.loc10_23.2, %.loc10_23.3)
// CHECK:STDOUT:   %.loc10_24: init %tuple.type.d07 = converted %.loc10_23.1, %.loc10_23.4
// CHECK:STDOUT:   %X.cpp_destructor.bound: <bound method> = bound_method %ax.var, constants.%X.cpp_destructor
// CHECK:STDOUT:   %X.cpp_destructor.call: init %empty_tuple.type = call %X.cpp_destructor.bound(%ax.var)
// CHECK:STDOUT:   return %.loc10_24 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.b7f625.1: type = class_type @Y.1 [concrete]
// CHECK:STDOUT:   %Y.elem.080: type = unbound_element_type %Y.b7f625.1, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.de4: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.de4) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet.de4 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Y.cpp_destructor.type.edb0f6.1: type = fn_type @Y.cpp_destructor.1 [concrete]
// CHECK:STDOUT:   %Y.cpp_destructor.4b6d21.1: %Y.cpp_destructor.type.edb0f6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.b7f625.2: type = class_type @Y.2 [concrete]
// CHECK:STDOUT:   %Y.elem.f4e: type = unbound_element_type %Y.b7f625.2, %Z [concrete]
// CHECK:STDOUT:   %Z.Z.type: type = fn_type @Z.Z [concrete]
// CHECK:STDOUT:   %Z.Z: %Z.Z.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.bf3: type = const_type %Z [concrete]
// CHECK:STDOUT:   %ptr.061: type = ptr_type %const.bf3 [concrete]
// CHECK:STDOUT:   %ptr.81c: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %Z__carbon_thunk.type: type = fn_type @Z__carbon_thunk [concrete]
// CHECK:STDOUT:   %Z__carbon_thunk: %Z__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.Op.type: type = fn_type @Z.Op [concrete]
// CHECK:STDOUT:   %Z.Op: %Z.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8e5: <witness> = custom_witness (%Z.Op), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.a4a: %Copy.type = facet_value %Z, (%custom_witness.8e5) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.465: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.a4a) [concrete]
// CHECK:STDOUT:   %.9a1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.465, %Copy.facet.a4a [concrete]
// CHECK:STDOUT:   %Y.cpp_destructor.type.edb0f6.2: type = fn_type @Y.cpp_destructor.2 [concrete]
// CHECK:STDOUT:   %Y.cpp_destructor.4b6d21.2: %Y.cpp_destructor.type.edb0f6.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.G: %G.type = import_ref Main//direct_import, G, loaded [concrete = constants.%G]
// CHECK:STDOUT:   %Main.H: %H.type = import_ref Main//direct_import, H, loaded [concrete = constants.%H]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Z.Z.decl: %Z.Z.type = fn_decl @Z.Z [concrete = constants.%Z.Z] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z__carbon_thunk.decl: %Z__carbon_thunk.type = fn_decl @Z__carbon_thunk [concrete = constants.%Z__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%Main.G [concrete = constants.%G]
// CHECK:STDOUT:   %.loc9_12.1: ref %Y.b7f625.1 = temporary_storage
// CHECK:STDOUT:   %G.call: init %Y.b7f625.1 to %.loc9_12.1 = call %G.ref()
// CHECK:STDOUT:   %.loc9_12.2: ref %Y.b7f625.1 = temporary %.loc9_12.1, %G.call
// CHECK:STDOUT:   %y.ref: %Y.elem.080 = name_ref y, @Y.1.%.1 [concrete = @Y.1.%.1]
// CHECK:STDOUT:   %.loc9_13.1: ref %i32 = class_element_access %.loc9_12.2, element0
// CHECK:STDOUT:   %.loc9_13.2: %i32 = acquire_value %.loc9_13.1
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc9_13.1: <bound method> = bound_method %.loc9_13.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_13.2: <bound method> = bound_method %.loc9_13.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc9_13.2(%.loc9_13.2)
// CHECK:STDOUT:   %Y.cpp_destructor.bound: <bound method> = bound_method %.loc9_12.2, constants.%Y.cpp_destructor.4b6d21.1
// CHECK:STDOUT:   %Y.cpp_destructor.call: init %empty_tuple.type = call %Y.cpp_destructor.bound(%.loc9_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseZ() -> out %return.param: %Z {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, imports.%Main.H [concrete = constants.%H]
// CHECK:STDOUT:   %.loc15_12.1: ref %Y.b7f625.2 = temporary_storage
// CHECK:STDOUT:   %H.call: init %Y.b7f625.2 to %.loc15_12.1 = call %H.ref()
// CHECK:STDOUT:   %.loc15_12.2: ref %Y.b7f625.2 = temporary %.loc15_12.1, %H.call
// CHECK:STDOUT:   %y.ref: %Y.elem.f4e = name_ref y, @Y.2.%.1 [concrete = @Y.2.%.1]
// CHECK:STDOUT:   %.loc15_13.1: ref %Z = class_element_access %.loc15_12.2, element0
// CHECK:STDOUT:   %.loc15_13.2: %Z = acquire_value %.loc15_13.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.9a1 = impl_witness_access constants.%custom_witness.8e5, element0 [concrete = constants.%Z.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc15_13.2, %impl.elem0
// CHECK:STDOUT:   %.loc15_13.3: ref %Z = temporary_storage
// CHECK:STDOUT:   %Op.ref: %Z.Z.type = name_ref Op, imports.%Z.Z.decl [concrete = constants.%Z.Z]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_13.4: ref %Z = value_as_ref %.loc15_13.2
// CHECK:STDOUT:   %addr.loc15_13.1: %ptr.81c = addr_of %.loc15_13.4
// CHECK:STDOUT:   %.loc15_13.5: %ptr.061 = as_compatible %addr.loc15_13.1
// CHECK:STDOUT:   %.loc15_13.6: %ptr.061 = converted %addr.loc15_13.1, %.loc15_13.5
// CHECK:STDOUT:   %addr.loc15_13.2: %ptr.81c = addr_of %.loc13_19.1
// CHECK:STDOUT:   %Z__carbon_thunk.call: init %empty_tuple.type = call imports.%Z__carbon_thunk.decl(%.loc15_13.6, %addr.loc15_13.2)
// CHECK:STDOUT:   %.loc15_13.7: init %Z to %.loc13_19.1 = mark_in_place_init %Z__carbon_thunk.call
// CHECK:STDOUT:   %Y.cpp_destructor.bound: <bound method> = bound_method %.loc15_12.2, constants.%Y.cpp_destructor.4b6d21.2
// CHECK:STDOUT:   %Y.cpp_destructor.call: init %empty_tuple.type = call %Y.cpp_destructor.bound(%.loc15_12.2)
// CHECK:STDOUT:   return %.loc15_13.7 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
