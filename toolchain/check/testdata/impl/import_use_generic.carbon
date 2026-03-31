// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_use_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_use_generic.carbon

// --- import_generic.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {}

interface I {
  fn F();
}

impl forall [T:! type] C(T) as I {
  fn F() {}
}

// --- fail_use_in_fn_decl.carbon

library "[[@TEST_NAME]]";

import library "import_generic";

var c: C({}) = {};
// We're just checking that this doesn't crash. It's not expected to compile.
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+8]]:11: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn G() -> c.(I.F)() {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+4]]:22: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: fn G() -> c.(I.F)() {}
// CHECK:STDERR:                      ^
// CHECK:STDERR:
fn G() -> c.(I.F)() {}
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+8]]:11: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn H() -> C({}).(I.F)() {}
// CHECK:STDERR:           ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+4]]:26: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: fn H() -> C({}).(I.F)() {}
// CHECK:STDERR:                          ^
// CHECK:STDERR:
fn H() -> C({}).(I.F)() {}

// CHECK:STDOUT: --- import_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.08c: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.705: %I.WithSelf.F.type.08c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F, @C.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.c16: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.708: %I.WithSelf.F.type.c16 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_15.2 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:     %C.loc10_27.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_27.1 (constants.%C)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc10_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.08c) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.705)] {} {}
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
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc10_15.2: type) {
// CHECK:STDOUT:   %T.loc10_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.1 (constants.%T)]
// CHECK:STDOUT:   %C.loc10_27.1: type = class_type @C, @C(%T.loc10_15.1) [symbolic = %C.loc10_27.1 (constants.%C)]
// CHECK:STDOUT:   %I.impl_witness.loc10_34.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc10_15.1) [symbolic = %I.impl_witness.loc10_34.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F, @C.as.I.impl(%T.loc10_15.1) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.%C.as.I.impl.F.type (%C.as.I.impl.F.type) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_27.2 as %I.ref {
// CHECK:STDOUT:     %C.as.I.impl.F.decl: @C.as.I.impl.%C.as.I.impl.F.type (%C.as.I.impl.F.type) = fn_decl @C.as.I.impl.F [symbolic = @C.as.I.impl.%C.as.I.impl.F (constants.%C.as.I.impl.F)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc10_34.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T) [symbolic = %I.impl_witness.loc10_34.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:     witness = %I.impl_witness.loc10_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F(@C.as.I.impl.%T.loc10_15.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.08c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc10_15.1 => constants.%T
// CHECK:STDOUT:   %C.loc10_27.1 => constants.%C
// CHECK:STDOUT:   %I.impl_witness.loc10_34.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type => constants.%C.as.I.impl.F.type
// CHECK:STDOUT:   %C.as.I.impl.F => constants.%C.as.I.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.c16
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.708
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_in_fn_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.850: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.526: type = pattern_type %C.850 [concrete]
// CHECK:STDOUT:   %C.val: %C.850 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.a1f: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.cc3: %I.WithSelf.F.type.a1f = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.972 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness.e33: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.ec1: type = fn_type @C.as.I.impl.F, @C.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.983: %C.as.I.impl.F.type.ec1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.impl_witness.79b: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.b66: type = fn_type @C.as.I.impl.F, @C.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.930: %C.as.I.impl.F.type.b66 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.850, (%I.impl_witness.79b) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.ca6: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ed7: %I.WithSelf.F.type.ca6 = struct_value () [concrete]
// CHECK:STDOUT:   %.864: type = fn_type_with_self_type %I.WithSelf.F.type.ca6, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F.930, @C.as.I.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//import_generic, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.I: type = import_ref Main//import_generic, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.820 = import_ref Main//import_generic, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//import_generic, loc4_10, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.a54: %I.assoc_type = import_ref Main//import_generic, loc7_9, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.F = import_ref Main//import_generic, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.2362f8.2: %I.type = import_ref Main//import_generic, loc6_13, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.c82 = import_ref Main//import_generic, loc6_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.972: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.a1f) = import_ref Main//import_generic, loc7_9, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.cc3)]
// CHECK:STDOUT:   %Main.import_ref.ce3: <witness> = import_ref Main//import_generic, loc10_34, loaded [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness.e33)]
// CHECK:STDOUT:   %Main.import_ref.870: @C.as.I.impl.%C.as.I.impl.F.type (%C.as.I.impl.F.type.ec1) = import_ref Main//import_generic, loc11_10, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F (constants.%C.as.I.impl.F.983)]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.870), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic, loc10_15, loaded [symbolic = @C.as.I.impl.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.c96: type = import_ref Main//import_generic, loc10_27, loaded [symbolic = @C.as.I.impl.%C (constants.%C.5a3)]
// CHECK:STDOUT:   %Main.import_ref.72a: type = import_ref Main//import_generic, loc10_32, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic, loc10_15, loaded [symbolic = @C.as.I.impl.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.526 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.526 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.850 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %C [concrete = constants.%C.850] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc6_11: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_12.2: type = converted %.loc6_11, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.850]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.850 = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {
// CHECK:STDOUT:     %c.ref: ref %C.850 = name_ref c, file.%c [concrete = file.%c.var]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, imports.%Main.import_ref.a54 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.loc16: %I.type = converted %c.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc25_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc25_15: type = converted %.loc25_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.850]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, imports.%Main.import_ref.a54 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %I.facet: %I.type = facet_value %C, (constants.%I.impl_witness.79b) [concrete = constants.%I.facet]
// CHECK:STDOUT:     %.loc25_16: %I.type = converted %C, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:     %impl.elem0: %.864 = impl_witness_access constants.%I.impl_witness.79b, element0 [concrete = constants.%C.as.I.impl.F.930]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.I.impl.F(constants.%empty_struct_type) [concrete = constants.%C.as.I.impl.F.specific_fn]
// CHECK:STDOUT:     %C.as.I.impl.F.call: init %empty_tuple.type = call %specific_fn()
// CHECK:STDOUT:     %.loc25_23: type = converted %C.as.I.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "import_generic.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c82
// CHECK:STDOUT:   .F = imports.%Main.import_ref.a54
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(imports.%Main.import_ref.b3bc94.3: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.5a3)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.e33)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F, @C.as.I.impl(%T) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.ec1)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.%C.as.I.impl.F.type (%C.as.I.impl.F.type.ec1) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.983)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.c96 as imports.%Main.import_ref.72a {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.ce3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.b3bc94.1: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.820
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.2362f8.2: %I.type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F(imports.%Main.import_ref.b3bc94.2: type) [from "import_generic.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_17.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_17.2: init %C.850 to file.%c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C.850 = converted %.loc6_17.1, %.loc6_17.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.a1f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.cc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.5a3
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.e33
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type => constants.%C.as.I.impl.F.type.ec1
// CHECK:STDOUT:   %C.as.I.impl.F => constants.%C.as.I.impl.F.983
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %C => constants.%C.850
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.79b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type => constants.%C.as.I.impl.F.type.b66
// CHECK:STDOUT:   %C.as.I.impl.F => constants.%C.as.I.impl.F.930
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.ca6
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ed7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
