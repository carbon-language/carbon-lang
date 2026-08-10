// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/final.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/final.carbon

// --- require_complete_impl.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

interface I { fn F(); }
interface J {}

final impl forall [T: J] T as I { fn F() {} }

// This is creating an indirection where the `final impl` requires an indirect
// instantiation by the caller.
fn UseI(generic T: I) { T.F(); }
fn UseJ(generic T: J) { UseI(T); }

class C {}
impl C as J {}
fn Use() { UseJ(C); }

// --- final_method.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface Final {
  final fn Method() { }
}

class C {}

impl C as Final { }

fn F(generic f: Final) {
  f.Method();
}

fn Run() {
  F(C);
}
//@dump-sem-ir-end

// --- final_method_for_cpp_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class C {};
''';

//@dump-sem-ir-begin
interface Final {
  final fn Method() { }
}

impl Cpp.C as Final { }

fn F(generic f: Final) {
  f.Method();
}

fn Run() {
  F(Cpp.C);
}
//@dump-sem-ir-end


// --- fail_final_method_multiple_impls.carbon

library "[[@TEST_NAME]]";

interface Final {
  final fn Method() { }
}

class C {}

// CHECK:STDERR: fail_final_method_multiple_impls.carbon:[[@LINE+7]]:1: error: attempted to implement `Method`, which is final in interface `Final` [ImplFinalFunction]
// CHECK:STDERR: impl C as Final {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_final_method_multiple_impls.carbon:[[@LINE-8]]:3: note: associated function Method declared here [AssociatedFunctionHere]
// CHECK:STDERR:   final fn Method() { }
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Final {
  fn Method() { }
}

// CHECK:STDOUT: --- require_complete_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.dcc: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ec6: %I.WithSelf.F.type.dcc = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2c5: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.patt.ac1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.e40: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.d90: type = facet_access_type %T.e40 [symbolic]
// CHECK:STDOUT:   %.d3d: <witness> = impl_self_witness %T.as_type.d90, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.b7c: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T.e40) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.f53: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.e40) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.305: %T.as_type.as.I.impl.F.type.f53 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.948: %I.type = facet_value %T.as_type.d90, (%I.impl_witness.b7c) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.117: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.948) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.caa: %I.WithSelf.F.type.117 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt.1df: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.dda: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %UseI.type: type = fn_type @UseI [concrete]
// CHECK:STDOUT:   %UseI: %UseI.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.418: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.dda) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.ca5: %I.WithSelf.F.type.418 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.b38: type = facet_access_type %T.dda [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.dda, @I [symbolic]
// CHECK:STDOUT:   %.81d: type = fn_type_with_self_type %I.WithSelf.F.type.418, %T.dda [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.81d = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.F(%T.dda) [symbolic]
// CHECK:STDOUT:   %UseJ.type: type = fn_type @UseJ [concrete]
// CHECK:STDOUT:   %UseJ: %UseJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %.062: require_specific_def_type = require_specific_def @T.as_type.as.I.impl(%T.e40) [symbolic]
// CHECK:STDOUT:   %UseI.specific_fn.68a: <specific function> = specific_function %UseI, @UseI(%I.facet.948) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.ffd: <witness> = impl_self_witness %C, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %UseJ.specific_fn: <specific function> = specific_function %UseJ, @UseJ(%J.facet) [concrete]
// CHECK:STDOUT:   %.d61: type = fn_type_with_self_type %I.WithSelf.F.type.117, %I.facet.948 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn.925: <specific function> = specific_function %T.as_type.as.I.impl.F.305, @T.as_type.as.I.impl.F(%T.e40) [symbolic]
// CHECK:STDOUT:   %.b7f: <witness> = impl_self_witness %C, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.324: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type.dbc: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.60c: %T.as_type.as.I.impl.F.type.dbc = struct_value () [concrete]
// CHECK:STDOUT:   %.3f3: require_specific_def_type = require_specific_def @T.as_type.as.I.impl(%J.facet) [concrete]
// CHECK:STDOUT:   %I.facet.a00: %I.type = facet_value %C, (%I.impl_witness.324) [concrete]
// CHECK:STDOUT:   %UseI.specific_fn.af1: <specific function> = specific_function %UseI, @UseI(%I.facet.a00) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.46b: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.a00) [concrete]
// CHECK:STDOUT:   %.c12: type = fn_type_with_self_type %I.WithSelf.F.type.46b, %I.facet.a00 [concrete]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.specific_fn.5d6: <specific function> = specific_function %T.as_type.as.I.impl.F.60c, @T.as_type.as.I.impl.F(%J.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .UseI = %UseI.decl
// CHECK:STDOUT:     .UseJ = %UseJ.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_21.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_21.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc7_21.2 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:     %T.as_type.loc7_26.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %.loc7_26: type = converted %T.ref, %T.as_type.loc7_26.2 [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_23: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_21.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @T.as_type.as.I.impl.%.loc7_26, @I [symbolic = @T.as_type.as.I.impl.%.loc7_33 (constants.%.d3d)]
// CHECK:STDOUT:   %UseI.decl: %UseI.type = fn_decl @UseI [concrete = constants.%UseI] {
// CHECK:STDOUT:     %T.patt.loc11_18.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_18.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_20: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_18.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc11_18.1 (constants.%T.dda)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseJ.decl: %UseJ.type = fn_decl @UseJ [concrete = constants.%UseJ] {
// CHECK:STDOUT:     %T.patt.loc12_18.1: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_18.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_20: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_18.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_18.1 (constants.%T.e40)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @C.as.J.impl.%C.ref, @J [concrete = constants.%.ffd]
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.dcc) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ec6)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic final impl @T.as_type.as.I.impl(%T.loc7_21.2: %J.type) {
// CHECK:STDOUT:   %T.patt.loc7_21.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_21.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   %T.loc7_21.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc7_21.1 (constants.%T.e40)]
// CHECK:STDOUT:   %T.as_type.loc7_26.1: type = facet_access_type %T.loc7_21.1 [symbolic = %T.as_type.loc7_26.1 (constants.%T.as_type.d90)]
// CHECK:STDOUT:   %.loc7_33: <witness> = impl_self_witness %T.as_type.loc7_26.1, @I [symbolic = %.loc7_33 (constants.%.d3d)]
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness.b7c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type: type = fn_type @T.as_type.as.I.impl.F, @T.as_type.as.I.impl(%T.loc7_21.1) [symbolic = %T.as_type.as.I.impl.F.type (constants.%T.as_type.as.I.impl.F.type.f53)]
// CHECK:STDOUT:   %T.as_type.as.I.impl.F: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.f53) = struct_value () [symbolic = %T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.305)]
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %.loc7_26 as %I.ref {
// CHECK:STDOUT:     %T.as_type.as.I.impl.F.decl: @T.as_type.as.I.impl.%T.as_type.as.I.impl.F.type (%T.as_type.as.I.impl.F.type.f53) = fn_decl @T.as_type.as.I.impl.F [symbolic = @T.as_type.as.I.impl.%T.as_type.as.I.impl.F (constants.%T.as_type.as.I.impl.F.305)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%T.as_type.as.I.impl.F.decl), @T.as_type.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_33.1: <witness> = impl_witness %I.impl_witness_table, @T.as_type.as.I.impl(constants.%T.e40) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness.b7c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %T.as_type.as.I.impl.F.decl
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_33.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as_type.as.I.impl.F(@T.as_type.as.I.impl.%T.loc7_21.2: %J.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseI(%T.loc11_18.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc11_18.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_18.2 (constants.%T.patt.1df)]
// CHECK:STDOUT:   %T.loc11_18.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc11_18.1 (constants.%T.dda)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_26.2: type = facet_access_type %T.loc11_18.1 [symbolic = %T.as_type.loc11_26.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.loc11_18.1) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.418)]
// CHECK:STDOUT:   %.loc11_26.2: type = fn_type_with_self_type %I.WithSelf.F.type, %T.loc11_18.1 [symbolic = %.loc11_26.2 (constants.%.81d)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_18.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc11_26.2: @UseI.%.loc11_26.2 (%.81d) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_26.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_26.2: <specific function> = specific_impl_function %impl.elem0.loc11_26.2, @I.WithSelf.F(%T.loc11_18.1) [symbolic = %specific_impl_fn.loc11_26.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc11_18.2 [symbolic = %T.loc11_18.1 (constants.%T.dda)]
// CHECK:STDOUT:     %T.as_type.loc11_26.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_26.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     %.loc11_26.1: type = converted %T.ref, %T.as_type.loc11_26.1 [symbolic = %T.as_type.loc11_26.2 (constants.%T.as_type.b38)]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc11_26.1: @UseI.%.loc11_26.2 (%.81d) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_26.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc11_26.1: <specific function> = specific_impl_function %impl.elem0.loc11_26.1, @I.WithSelf.F(constants.%T.dda) [symbolic = %specific_impl_fn.loc11_26.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc11_26.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseJ(%T.loc12_18.2: %J.type) {
// CHECK:STDOUT:   %T.patt.loc12_18.2: %pattern_type.2c5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_18.2 (constants.%T.patt.ac1)]
// CHECK:STDOUT:   %T.loc12_18.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc12_18.1 (constants.%T.e40)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_31.2: type = facet_access_type %T.loc12_18.1 [symbolic = %T.as_type.loc12_31.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:   %.loc12_31.2: require_specific_def_type = require_specific_def @T.as_type.as.I.impl(%T.loc12_18.1) [symbolic = %.loc12_31.2 (constants.%.062)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as_type.as.I.impl.%I.impl_witness_table, @T.as_type.as.I.impl(%T.loc12_18.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.b7c)]
// CHECK:STDOUT:   %I.facet.loc12_31.2: %I.type = facet_value %T.as_type.loc12_31.2, (%I.impl_witness) [symbolic = %I.facet.loc12_31.2 (constants.%I.facet.948)]
// CHECK:STDOUT:   %UseI.specific_fn.loc12_25.2: <specific function> = specific_function constants.%UseI, @UseI(%I.facet.loc12_31.2) [symbolic = %UseI.specific_fn.loc12_25.2 (constants.%UseI.specific_fn.68a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %UseI.ref: %UseI.type = name_ref UseI, file.%UseI.decl [concrete = constants.%UseI]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc12_18.2 [symbolic = %T.loc12_18.1 (constants.%T.e40)]
// CHECK:STDOUT:     %T.as_type.loc12_31.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_31.2 (constants.%T.as_type.d90)]
// CHECK:STDOUT:     %I.facet.loc12_31.1: %I.type = facet_value %T.as_type.loc12_31.1, (constants.%I.impl_witness.b7c) [symbolic = %I.facet.loc12_31.2 (constants.%I.facet.948)]
// CHECK:STDOUT:     %.loc12_31.1: %I.type = converted %T.ref, %I.facet.loc12_31.1 [symbolic = %I.facet.loc12_31.2 (constants.%I.facet.948)]
// CHECK:STDOUT:     %UseI.specific_fn.loc12_25.1: <specific function> = specific_function %UseI.ref, @UseI(constants.%I.facet.948) [symbolic = %UseI.specific_fn.loc12_25.2 (constants.%UseI.specific_fn.68a)]
// CHECK:STDOUT:     %UseI.call: init %empty_tuple.type = call %UseI.specific_fn.loc12_25.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %UseJ.ref: %UseJ.type = name_ref UseJ, file.%UseJ.decl [concrete = constants.%UseJ]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C.ref, (constants.%J.impl_witness) [concrete = constants.%J.facet]
// CHECK:STDOUT:   %.loc16: %J.type = converted %C.ref, %J.facet [concrete = constants.%J.facet]
// CHECK:STDOUT:   %UseJ.specific_fn: <specific function> = specific_function %UseJ.ref, @UseJ(constants.%J.facet) [concrete = constants.%UseJ.specific_fn]
// CHECK:STDOUT:   %UseJ.call: init %empty_tuple.type = call %UseJ.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.dcc
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ec6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%T.e40) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%T.e40
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%T.as_type.d90
// CHECK:STDOUT:   %.loc7_33 => constants.%.d3d
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.b7c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.f53
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.305
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%T.e40) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.948) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.948
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.117
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.caa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.948) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%T.dda) {
// CHECK:STDOUT:   %T.patt.loc11_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc11_18.1 => constants.%T.dda
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.dda) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T.dda
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.418
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ca5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T.dda) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseJ(constants.%T.e40) {
// CHECK:STDOUT:   %T.patt.loc12_18.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc12_18.1 => constants.%T.e40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%I.facet.948) {
// CHECK:STDOUT:   %T.patt.loc11_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc11_18.1 => constants.%I.facet.948
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_26.2 => constants.%T.as_type.d90
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.117
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.d61
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.b7c
// CHECK:STDOUT:   %impl.elem0.loc11_26.2 => constants.%T.as_type.as.I.impl.F.305
// CHECK:STDOUT:   %specific_impl_fn.loc11_26.2 => constants.%T.as_type.as.I.impl.F.specific_fn.925
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseJ(constants.%J.facet) {
// CHECK:STDOUT:   %T.patt.loc12_18.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc12_18.1 => constants.%J.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_31.2 => constants.%C
// CHECK:STDOUT:   %.loc12_31.2 => constants.%.3f3
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.324
// CHECK:STDOUT:   %I.facet.loc12_31.2 => constants.%I.facet.a00
// CHECK:STDOUT:   %UseI.specific_fn.loc12_25.2 => constants.%UseI.specific_fn.af1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl(constants.%J.facet) {
// CHECK:STDOUT:   %T.patt.loc7_21.2 => constants.%T.patt.ac1
// CHECK:STDOUT:   %T.loc7_21.1 => constants.%J.facet
// CHECK:STDOUT:   %T.as_type.loc7_26.1 => constants.%C
// CHECK:STDOUT:   %.loc7_33 => constants.%.b7f
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness.324
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.as.I.impl.F.type => constants.%T.as_type.as.I.impl.F.type.dbc
// CHECK:STDOUT:   %T.as_type.as.I.impl.F => constants.%T.as_type.as.I.impl.F.60c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseI(constants.%I.facet.a00) {
// CHECK:STDOUT:   %T.patt.loc11_18.2 => constants.%T.patt.1df
// CHECK:STDOUT:   %T.loc11_18.1 => constants.%I.facet.a00
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_26.2 => constants.%C
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.46b
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.c12
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.324
// CHECK:STDOUT:   %impl.elem0.loc11_26.2 => constants.%T.as_type.as.I.impl.F.60c
// CHECK:STDOUT:   %specific_impl_fn.loc11_26.2 => constants.%T.as_type.as.I.impl.F.specific_fn.5d6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.a00) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.a00) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.I.impl.F(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Final.type: type = facet_type <@Final> [concrete]
// CHECK:STDOUT:   %Self: %Final.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Final.WithSelf.Method.type.cfa: type = fn_type @Final.WithSelf.Method, @Final.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.bf0: %Final.WithSelf.Method.type.cfa = struct_value () [symbolic]
// CHECK:STDOUT:   %Final.assoc_type: type = assoc_entity_type @Final [concrete]
// CHECK:STDOUT:   %assoc0: %Final.assoc_type = assoc_entity element0, @Final.WithSelf.%Final.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.966: <witness> = impl_self_witness %C, @Final [concrete]
// CHECK:STDOUT:   %Final.impl_witness: <witness> = impl_witness @C.as.Final.impl.%Final.impl_witness_table [concrete]
// CHECK:STDOUT:   %Final.facet: %Final.type = facet_value %C, (%Final.impl_witness) [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.type.877: type = fn_type @Final.WithSelf.Method, @Final.WithSelf(%Final.facet) [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.2df: %Final.WithSelf.Method.type.877 = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Final.type [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type = symbolic_binding_pattern f, 0 [symbolic]
// CHECK:STDOUT:   %f: %Final.type = symbolic_binding f, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.type.470: type = fn_type @Final.WithSelf.Method, @Final.WithSelf(%f) [symbolic]
// CHECK:STDOUT:   %Final.WithSelf.Method.9a3: %Final.WithSelf.Method.type.470 = struct_value () [symbolic]
// CHECK:STDOUT:   %f.as_type: type = facet_access_type %f [symbolic]
// CHECK:STDOUT:   %Final.lookup_impl_witness: <witness> = lookup_impl_witness %f, @Final [symbolic]
// CHECK:STDOUT:   %.898: type = fn_type_with_self_type %Final.WithSelf.Method.type.470, %f [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.898 = impl_witness_access %Final.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Final.WithSelf.Method(%f) [symbolic]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Final.facet) [concrete]
// CHECK:STDOUT:   %.090: type = fn_type_with_self_type %Final.WithSelf.Method.type.877, %Final.facet [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.specific_fn: <specific function> = specific_function %Final.WithSelf.Method.bf0, @Final.WithSelf.Method(%Self) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Final.decl: type = interface_decl @Final [concrete = constants.%Final.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Final.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Final.ref: type = name_ref Final, file.%Final.decl [concrete = constants.%Final.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @C.as.Final.impl.%C.ref, @Final [concrete = constants.%.966]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %f.patt.loc13_15.1: %pattern_type = symbolic_binding_pattern f, 0 [symbolic = %f.patt.loc13_15.2 (constants.%f.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %Final.ref [concrete = constants.%Final.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Final.ref: type = name_ref Final, file.%Final.decl [concrete = constants.%Final.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %f.loc13_15.2: %Final.type = symbolic_binding f, 0 [symbolic = %f.loc13_15.1 (constants.%f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Final {
// CHECK:STDOUT:   %Self: %Final.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Final.WithSelf.decl = interface_with_self_decl @Final [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Final.WithSelf.Method.decl: @Final.WithSelf.%Final.WithSelf.Method.type (%Final.WithSelf.Method.type.cfa) = fn_decl @Final.WithSelf.Method [symbolic = @Final.WithSelf.%Final.WithSelf.Method (constants.%Final.WithSelf.Method.bf0)] {} {}
// CHECK:STDOUT:   %assoc0: %Final.assoc_type = assoc_entity element0, %Final.WithSelf.Method.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Method = @Final.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Final.WithSelf.%Final.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Final.impl: %C.ref as %Final.ref {
// CHECK:STDOUT:   %Final.impl_witness_table = impl_witness_table (@Final.WithSelf.%Final.WithSelf.Method.decl), @C.as.Final.impl [concrete]
// CHECK:STDOUT:   %Final.impl_witness: <witness> = impl_witness %Final.impl_witness_table [concrete = constants.%Final.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Method = <poisoned>
// CHECK:STDOUT:   extend %Final.ref
// CHECK:STDOUT:   witness = %Final.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Final.WithSelf.Method(@Final.%Self: %Final.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%f.loc13_15.2: %Final.type) {
// CHECK:STDOUT:   %f.patt.loc13_15.2: %pattern_type = symbolic_binding_pattern f, 0 [symbolic = %f.patt.loc13_15.2 (constants.%f.patt)]
// CHECK:STDOUT:   %f.loc13_15.1: %Final.type = symbolic_binding f, 0 [symbolic = %f.loc13_15.1 (constants.%f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %f.as_type.loc14_4.2: type = facet_access_type %f.loc13_15.1 [symbolic = %f.as_type.loc14_4.2 (constants.%f.as_type)]
// CHECK:STDOUT:   %Final.WithSelf.Method.type: type = fn_type @Final.WithSelf.Method, @Final.WithSelf(%f.loc13_15.1) [symbolic = %Final.WithSelf.Method.type (constants.%Final.WithSelf.Method.type.470)]
// CHECK:STDOUT:   %.loc14_4.2: type = fn_type_with_self_type %Final.WithSelf.Method.type, %f.loc13_15.1 [symbolic = %.loc14_4.2 (constants.%.898)]
// CHECK:STDOUT:   %Final.lookup_impl_witness: <witness> = lookup_impl_witness %f.loc13_15.1, @Final [symbolic = %Final.lookup_impl_witness (constants.%Final.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @F.%.loc14_4.2 (%.898) = impl_witness_access %Final.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @Final.WithSelf.Method(%f.loc13_15.1) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %f.ref: %Final.type = name_ref f, %f.loc13_15.2 [symbolic = %f.loc13_15.1 (constants.%f)]
// CHECK:STDOUT:     %f.as_type.loc14_4.1: type = facet_access_type %f.ref [symbolic = %f.as_type.loc14_4.2 (constants.%f.as_type)]
// CHECK:STDOUT:     %.loc14_4.1: type = converted %f.ref, %f.as_type.loc14_4.1 [symbolic = %f.as_type.loc14_4.2 (constants.%f.as_type)]
// CHECK:STDOUT:     %Method.ref: %Final.assoc_type = name_ref Method, @Final.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @F.%.loc14_4.2 (%.898) = impl_witness_access constants.%Final.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @Final.WithSelf.Method(constants.%f) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Final.WithSelf.Method.call: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Final.facet: %Final.type = facet_value %C.ref, (constants.%Final.impl_witness) [concrete = constants.%Final.facet]
// CHECK:STDOUT:   %.loc18: %Final.type = converted %C.ref, %Final.facet [concrete = constants.%Final.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Final.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Final.WithSelf.Method.type => constants.%Final.WithSelf.Method.type.cfa
// CHECK:STDOUT:   %Final.WithSelf.Method => constants.%Final.WithSelf.Method.bf0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf.Method(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf(constants.%Final.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Final.facet
// CHECK:STDOUT:   %Final.WithSelf.Method.type => constants.%Final.WithSelf.Method.type.877
// CHECK:STDOUT:   %Final.WithSelf.Method => constants.%Final.WithSelf.Method.2df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf.Method(constants.%Final.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%f) {
// CHECK:STDOUT:   %f.patt.loc13_15.2 => constants.%f.patt
// CHECK:STDOUT:   %f.loc13_15.1 => constants.%f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf(constants.%f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%f
// CHECK:STDOUT:   %Final.WithSelf.Method.type => constants.%Final.WithSelf.Method.type.470
// CHECK:STDOUT:   %Final.WithSelf.Method => constants.%Final.WithSelf.Method.9a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf.Method(constants.%f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Final.facet) {
// CHECK:STDOUT:   %f.patt.loc13_15.2 => constants.%f.patt
// CHECK:STDOUT:   %f.loc13_15.1 => constants.%Final.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %f.as_type.loc14_4.2 => constants.%C
// CHECK:STDOUT:   %Final.WithSelf.Method.type => constants.%Final.WithSelf.Method.type.877
// CHECK:STDOUT:   %.loc14_4.2 => constants.%.090
// CHECK:STDOUT:   %Final.lookup_impl_witness => constants.%Final.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc14_4.2 => constants.%Final.WithSelf.Method.bf0
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2 => constants.%Final.WithSelf.Method.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_method_for_cpp_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Final.type: type = facet_type <@Final> [concrete]
// CHECK:STDOUT:   %Self: %Final.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Final.WithSelf.Method.type.cfa: type = fn_type @Final.WithSelf.Method, @Final.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.bf0: %Final.WithSelf.Method.type.cfa = struct_value () [symbolic]
// CHECK:STDOUT:   %Final.assoc_type: type = assoc_entity_type @Final [concrete]
// CHECK:STDOUT:   %assoc0: %Final.assoc_type = assoc_entity element0, @Final.WithSelf.%Final.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.16c: <witness> = impl_self_witness %C, @Final [concrete]
// CHECK:STDOUT:   %Final.impl_witness: <witness> = impl_witness @C.as.Final.impl.%Final.impl_witness_table [concrete]
// CHECK:STDOUT:   %Final.facet: %Final.type = facet_value %C, (%Final.impl_witness) [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.type.d8a: type = fn_type @Final.WithSelf.Method, @Final.WithSelf(%Final.facet) [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.410: %Final.WithSelf.Method.type.d8a = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Final.type [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type = symbolic_binding_pattern f, 0 [symbolic]
// CHECK:STDOUT:   %f: %Final.type = symbolic_binding f, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.type.470: type = fn_type @Final.WithSelf.Method, @Final.WithSelf(%f) [symbolic]
// CHECK:STDOUT:   %Final.WithSelf.Method.9a3: %Final.WithSelf.Method.type.470 = struct_value () [symbolic]
// CHECK:STDOUT:   %f.as_type: type = facet_access_type %f [symbolic]
// CHECK:STDOUT:   %Final.lookup_impl_witness: <witness> = lookup_impl_witness %f, @Final [symbolic]
// CHECK:STDOUT:   %.898: type = fn_type_with_self_type %Final.WithSelf.Method.type.470, %f [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.898 = impl_witness_access %Final.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Final.WithSelf.Method(%f) [symbolic]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Final.facet) [concrete]
// CHECK:STDOUT:   %.980: type = fn_type_with_self_type %Final.WithSelf.Method.type.d8a, %Final.facet [concrete]
// CHECK:STDOUT:   %Final.WithSelf.Method.specific_fn: <specific function> = specific_function %Final.WithSelf.Method.bf0, @Final.WithSelf.Method(%Self) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Final.decl: type = interface_decl @Final [concrete = constants.%Final.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Final.impl [concrete] {} {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Final.ref: type = name_ref Final, file.%Final.decl [concrete = constants.%Final.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.Final.impl.%C.ref, @Final [concrete = constants.%.16c]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %f.patt.loc15_15.1: %pattern_type = symbolic_binding_pattern f, 0 [symbolic = %f.patt.loc15_15.2 (constants.%f.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %Final.ref [concrete = constants.%Final.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Final.ref: type = name_ref Final, file.%Final.decl [concrete = constants.%Final.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %f.loc15_15.2: %Final.type = symbolic_binding f, 0 [symbolic = %f.loc15_15.1 (constants.%f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Final {
// CHECK:STDOUT:   %Self: %Final.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Final.WithSelf.decl = interface_with_self_decl @Final [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Final.WithSelf.Method.decl: @Final.WithSelf.%Final.WithSelf.Method.type (%Final.WithSelf.Method.type.cfa) = fn_decl @Final.WithSelf.Method [symbolic = @Final.WithSelf.%Final.WithSelf.Method (constants.%Final.WithSelf.Method.bf0)] {} {}
// CHECK:STDOUT:   %assoc0: %Final.assoc_type = assoc_entity element0, %Final.WithSelf.Method.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Method = @Final.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Final.WithSelf.%Final.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Final.impl: %C.ref as %Final.ref {
// CHECK:STDOUT:   %Final.impl_witness_table = impl_witness_table (@Final.WithSelf.%Final.WithSelf.Method.decl), @C.as.Final.impl [concrete]
// CHECK:STDOUT:   %Final.impl_witness: <witness> = impl_witness %Final.impl_witness_table [concrete = constants.%Final.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Method = <poisoned>
// CHECK:STDOUT:   extend %Final.ref
// CHECK:STDOUT:   witness = %Final.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Final.WithSelf.Method(@Final.%Self: %Final.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%f.loc15_15.2: %Final.type) {
// CHECK:STDOUT:   %f.patt.loc15_15.2: %pattern_type = symbolic_binding_pattern f, 0 [symbolic = %f.patt.loc15_15.2 (constants.%f.patt)]
// CHECK:STDOUT:   %f.loc15_15.1: %Final.type = symbolic_binding f, 0 [symbolic = %f.loc15_15.1 (constants.%f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %f.as_type.loc16_4.2: type = facet_access_type %f.loc15_15.1 [symbolic = %f.as_type.loc16_4.2 (constants.%f.as_type)]
// CHECK:STDOUT:   %Final.WithSelf.Method.type: type = fn_type @Final.WithSelf.Method, @Final.WithSelf(%f.loc15_15.1) [symbolic = %Final.WithSelf.Method.type (constants.%Final.WithSelf.Method.type.470)]
// CHECK:STDOUT:   %.loc16_4.2: type = fn_type_with_self_type %Final.WithSelf.Method.type, %f.loc15_15.1 [symbolic = %.loc16_4.2 (constants.%.898)]
// CHECK:STDOUT:   %Final.lookup_impl_witness: <witness> = lookup_impl_witness %f.loc15_15.1, @Final [symbolic = %Final.lookup_impl_witness (constants.%Final.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @F.%.loc16_4.2 (%.898) = impl_witness_access %Final.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @Final.WithSelf.Method(%f.loc15_15.1) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %f.ref: %Final.type = name_ref f, %f.loc15_15.2 [symbolic = %f.loc15_15.1 (constants.%f)]
// CHECK:STDOUT:     %f.as_type.loc16_4.1: type = facet_access_type %f.ref [symbolic = %f.as_type.loc16_4.2 (constants.%f.as_type)]
// CHECK:STDOUT:     %.loc16_4.1: type = converted %f.ref, %f.as_type.loc16_4.1 [symbolic = %f.as_type.loc16_4.2 (constants.%f.as_type)]
// CHECK:STDOUT:     %Method.ref: %Final.assoc_type = name_ref Method, @Final.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @F.%.loc16_4.2 (%.898) = impl_witness_access constants.%Final.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_4.1: <specific function> = specific_impl_function %impl.elem0.loc16_4.1, @Final.WithSelf.Method(constants.%f) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Final.WithSelf.Method.call: init %empty_tuple.type = call %specific_impl_fn.loc16_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Final.facet: %Final.type = facet_value %C.ref, (constants.%Final.impl_witness) [concrete = constants.%Final.facet]
// CHECK:STDOUT:   %.loc20: %Final.type = converted %C.ref, %Final.facet [concrete = constants.%Final.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Final.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Final.WithSelf.Method.type => constants.%Final.WithSelf.Method.type.cfa
// CHECK:STDOUT:   %Final.WithSelf.Method => constants.%Final.WithSelf.Method.bf0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf.Method(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf(constants.%Final.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Final.facet
// CHECK:STDOUT:   %Final.WithSelf.Method.type => constants.%Final.WithSelf.Method.type.d8a
// CHECK:STDOUT:   %Final.WithSelf.Method => constants.%Final.WithSelf.Method.410
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf.Method(constants.%Final.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%f) {
// CHECK:STDOUT:   %f.patt.loc15_15.2 => constants.%f.patt
// CHECK:STDOUT:   %f.loc15_15.1 => constants.%f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf(constants.%f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%f
// CHECK:STDOUT:   %Final.WithSelf.Method.type => constants.%Final.WithSelf.Method.type.470
// CHECK:STDOUT:   %Final.WithSelf.Method => constants.%Final.WithSelf.Method.9a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Final.WithSelf.Method(constants.%f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Final.facet) {
// CHECK:STDOUT:   %f.patt.loc15_15.2 => constants.%f.patt
// CHECK:STDOUT:   %f.loc15_15.1 => constants.%Final.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %f.as_type.loc16_4.2 => constants.%C
// CHECK:STDOUT:   %Final.WithSelf.Method.type => constants.%Final.WithSelf.Method.type.d8a
// CHECK:STDOUT:   %.loc16_4.2 => constants.%.980
// CHECK:STDOUT:   %Final.lookup_impl_witness => constants.%Final.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc16_4.2 => constants.%Final.WithSelf.Method.bf0
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2 => constants.%Final.WithSelf.Method.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
