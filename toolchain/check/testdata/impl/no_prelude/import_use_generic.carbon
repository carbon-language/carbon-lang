// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_use_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_use_generic.carbon

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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.40c: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.071: %F.type.40c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc10_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc10_27.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_27.2 (constants.%C)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc10_14.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %C.loc10_27.2: type = class_type @C, @C(%T.loc10_14.2) [symbolic = %C.loc10_27.2 (constants.%C)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%T.loc10_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc10_14.2) [symbolic = %F.type (constants.%F.type.40c)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.40c) = struct_value () [symbolic = %F (constants.%F.071)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_27.1 as %I.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.40c) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.071)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc10_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc10_27.2 => constants.%C
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.40c
// CHECK:STDOUT:   %F => constants.%F.071
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T.loc10_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc10_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_in_fn_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C.7a7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.777 [concrete]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %impl_witness.681: <witness> = impl_witness (imports.%Main.import_ref.e2c), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.40c: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.071: %F.type.40c = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.02b: <witness> = impl_witness (imports.%Main.import_ref.e2c), @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %impl_witness.bbe: <witness> = impl_witness (imports.%Main.import_ref.c57), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.4a4: type = fn_type @F.2, @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %F.9e6: %F.type.4a4 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.7a7, (%impl_witness.02b) [concrete]
// CHECK:STDOUT:   %.dd0: type = fn_type_with_self_type %F.type.cf0, %I.facet [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.9e6, @F.2(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//import_generic, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.I: type = import_ref Main//import_generic, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//import_generic, loc4_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.4c0 = import_ref Main//import_generic, inst23 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//import_generic, inst29 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bcb: %I.assoc_type = import_ref Main//import_generic, loc7_9, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.F: %F.type.cf0 = import_ref Main//import_generic, F, loaded [concrete = constants.%F.bc6]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//import_generic, inst29 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.d91: <witness> = import_ref Main//import_generic, loc10_34, loaded [symbolic = @impl.%impl_witness (constants.%impl_witness.bbe)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//import_generic, loc10_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.499: type = import_ref Main//import_generic, loc10_27, loaded [symbolic = @impl.%C (constants.%C.f2e)]
// CHECK:STDOUT:   %Main.import_ref.301: type = import_ref Main//import_generic, loc10_32, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.e2c: @impl.%F.type (%F.type.40c) = import_ref Main//import_generic, loc11_10, loaded [symbolic = @impl.%F (constants.%F.071)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//import_generic, loc10_14, loaded [symbolic = @impl.%T (constants.%T)]
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
// CHECK:STDOUT:     %c.patt: %C.7a7 = binding_pattern c
// CHECK:STDOUT:     %.loc6_1: %C.7a7 = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.7a7 = var c
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %C [concrete = constants.%C.7a7] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc6_11: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_12.2: type = converted %.loc6_11, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.7a7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.7a7 = bind_name c, %c.var
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.ref: ref %C.7a7 = name_ref c, file.%c
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, imports.%Main.import_ref.bcb [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.loc16: %I.type = converted %c.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc25_14: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc25_15: type = converted %.loc25_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.7a7]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, imports.%Main.import_ref.bcb [concrete = constants.%assoc0]
// CHECK:STDOUT:     %I.facet: %I.type = facet_value constants.%C.7a7, (constants.%impl_witness.02b) [concrete = constants.%I.facet]
// CHECK:STDOUT:     %.loc25_16: %I.type = converted %C, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:     %as_wit.iface0: <witness> = facet_access_witness %.loc25_16, element0 [concrete = constants.%impl_witness.02b]
// CHECK:STDOUT:     %impl.elem0: %.dd0 = impl_witness_access %as_wit.iface0, element0 [concrete = constants.%F.9e6]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%empty_struct_type) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %specific_fn()
// CHECK:STDOUT:     %.loc25_23.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:     %.loc25_23.2: ref %empty_tuple.type = temporary %.loc25_23.1, %F.call
// CHECK:STDOUT:     %.loc25_23.3: type = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "import_generic.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%Main.import_ref.bcb
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Main.import_ref.5ab3ec.2: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.e2c), @impl(%T) [symbolic = %impl_witness (constants.%impl_witness.681)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T) [symbolic = %F.type (constants.%F.type.40c)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.40c) = struct_value () [symbolic = %F (constants.%F.071)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.499 as imports.%Main.import_ref.301 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.d91
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.5ab3ec.1: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.4c0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5dd: %I.type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(imports.%Main.import_ref.5ab3ec.3: type) [from "import_generic.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_17.2: init %C.7a7 = class_init (), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C.7a7 = converted %.loc6_17.1, %.loc6_17.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %C => constants.%C.f2e
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.681
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.40c
// CHECK:STDOUT:   %F => constants.%F.071
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt => constants.%empty_struct_type
// CHECK:STDOUT:   %C => constants.%C.7a7
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.02b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.4a4
// CHECK:STDOUT:   %F => constants.%F.9e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%empty_struct_type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
