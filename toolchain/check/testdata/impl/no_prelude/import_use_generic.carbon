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
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+12]]:14: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn G() -> c.(I.F)() {}
// CHECK:STDERR:              ^~~
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: import_generic.carbon:6:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+4]]:22: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: fn G() -> c.(I.F)() {}
// CHECK:STDERR:                      ^
// CHECK:STDERR:
fn G() -> c.(I.F)() {}
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE+12]]:18: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn H() -> C({}).(I.F)() {}
// CHECK:STDERR:                  ^~~
// CHECK:STDERR: fail_use_in_fn_decl.carbon:[[@LINE-20]]:1: in import [InImport]
// CHECK:STDERR: import_generic.carbon:6:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I {
// CHECK:STDERR: ^~~~~~~~~~~~~
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
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C.7a7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//import_generic, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.I: type = import_ref Main//import_generic, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.5ab: type = import_ref Main//import_generic, loc4_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.4c0 = import_ref Main//import_generic, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//import_generic, inst30 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1e0 = import_ref Main//import_generic, loc7_9, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//import_generic, F, unloaded
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
// CHECK:STDOUT:     %F.ref: <error> = name_ref F, <error> [concrete = <error>]
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
// CHECK:STDOUT:     %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "import_generic.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%Main.import_ref.1e0
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.5ab: type) [from "import_generic.carbon"] {
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
// CHECK:STDOUT: fn @G() -> <error> {
// CHECK:STDOUT: !entry:
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
