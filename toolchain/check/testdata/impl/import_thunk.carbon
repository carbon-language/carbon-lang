// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_thunk.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F(x: {});
}

// --- b.carbon

library "[[@TEST_NAME]]";
import library "a";

class C(X:! ()) {}

impl forall [Y:! ()] C(Y) as I {
  fn F(x: C(Y)) {}
}

// --- c.carbon

library "[[@TEST_NAME]]";
import library "a";
import library "b";

fn G() {
  C(()).(I.F)({});
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc5_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn(%x.param: %empty_struct_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %X: %empty_tuple.type = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.13320f.1: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic]
// CHECK:STDOUT:   %C.13320f.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @impl(%Y) [symbolic]
// CHECK:STDOUT:   %pattern_type.ccc: type = pattern_type %C.13320f.2 [symbolic]
// CHECK:STDOUT:   %F.type.0daaa1.1: type = fn_type @F.2, @impl(%Y) [symbolic]
// CHECK:STDOUT:   %F.49c1ac.1: %F.type.0daaa1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.13320f.2, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %F.type.0daaa1.2: type = fn_type @F.3, @impl(%Y) [symbolic]
// CHECK:STDOUT:   %F.49c1ac.2: %F.type.0daaa1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.13320f.2 [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.49c1ac.1, @F.2(%Y) [symbolic]
// CHECK:STDOUT:   %C.val: %C.13320f.2 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//a, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.507 = import_ref Main//a, loc5_14, unloaded
// CHECK:STDOUT:   %Main.F: %F.type.cf0 = import_ref Main//a, F, loaded [concrete = constants.%F.bc6]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//a, inst19 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %X.patt: %pattern_type.cb1 = symbolic_binding_pattern X, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc5_14.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc5_14.3: type = converted %.loc5_14.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc5_9.1: %empty_tuple.type = bind_symbolic_name X, 0 [symbolic = %X.loc5_9.2 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Y.ref: %empty_tuple.type = name_ref Y, %Y.loc7_14.1 [symbolic = %Y.loc7_14.2 (constants.%Y)]
// CHECK:STDOUT:     %C.loc7_25.1: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc7_25.2 (constants.%C.13320f.2)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc7_19.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc7_19.3: type = converted %.loc7_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.loc7_14.1: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic = %Y.loc7_14.2 (constants.%Y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.%F.decl.loc8_17.2), @impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @impl(constants.%Y) [symbolic = @impl.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%Main.import_ref.507
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%Y.loc7_14.1: %empty_tuple.type) {
// CHECK:STDOUT:   %Y.loc7_14.2: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic = %Y.loc7_14.2 (constants.%Y)]
// CHECK:STDOUT:   %C.loc7_25.2: type = class_type @C, @C(%Y.loc7_14.2) [symbolic = %C.loc7_25.2 (constants.%C.13320f.2)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @impl(%Y.loc7_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type.loc8_17.1: type = fn_type @F.2, @impl(%Y.loc7_14.2) [symbolic = %F.type.loc8_17.1 (constants.%F.type.0daaa1.1)]
// CHECK:STDOUT:   %F.loc8_17.1: @impl.%F.type.loc8_17.1 (%F.type.0daaa1.1) = struct_value () [symbolic = %F.loc8_17.1 (constants.%F.49c1ac.1)]
// CHECK:STDOUT:   %F.type.loc8_17.2: type = fn_type @F.3, @impl(%Y.loc7_14.2) [symbolic = %F.type.loc8_17.2 (constants.%F.type.0daaa1.2)]
// CHECK:STDOUT:   %F.loc8_17.2: @impl.%F.type.loc8_17.2 (%F.type.0daaa1.2) = struct_value () [symbolic = %F.loc8_17.2 (constants.%F.49c1ac.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_25.1 as %I.ref {
// CHECK:STDOUT:     %F.decl.loc8_17.1: @impl.%F.type.loc8_17.1 (%F.type.0daaa1.1) = fn_decl @F.2 [symbolic = @impl.%F.loc8_17.1 (constants.%F.49c1ac.1)] {
// CHECK:STDOUT:       %x.patt: @F.2.%pattern_type (%pattern_type.ccc) = binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @F.2.%pattern_type (%pattern_type.ccc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @F.2.%C.loc8_14.1 (%C.13320f.2) = value_param call_param0
// CHECK:STDOUT:       %.loc8: type = splice_block %C.loc8_14.2 [symbolic = %C.loc8_14.1 (constants.%C.13320f.2)] {
// CHECK:STDOUT:         %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:         %Y.ref: %empty_tuple.type = name_ref Y, @impl.%Y.loc7_14.1 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:         %C.loc8_14.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc8_14.1 (constants.%C.13320f.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @F.2.%C.loc8_14.1 (%C.13320f.2) = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.decl.loc8_17.2: @impl.%F.type.loc8_17.2 (%F.type.0daaa1.2) = fn_decl @F.3 [symbolic = @impl.%F.loc8_17.2 (constants.%F.49c1ac.2)] {
// CHECK:STDOUT:       %x.patt: %pattern_type.a96 = binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: %pattern_type.a96 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:       %x: %empty_struct_type = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .Y = <poisoned>
// CHECK:STDOUT:     .F = %F.decl.loc8_17.1
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%X.loc5_9.1: %empty_tuple.type) {
// CHECK:STDOUT:   %X.loc5_9.2: %empty_tuple.type = bind_symbolic_name X, 0 [symbolic = %X.loc5_9.2 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.13320f.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5dd: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%Y.loc7_14.1: %empty_tuple.type) {
// CHECK:STDOUT:   %Y: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.loc8_14.1: type = class_type @C, @C(%Y) [symbolic = %C.loc8_14.1 (constants.%C.13320f.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc8_14.1 [symbolic = %pattern_type (constants.%pattern_type.ccc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc8_14.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.2.%C.loc8_14.1 (%C.13320f.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.3(@impl.%Y.loc7_14.1: %empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%Y) [symbolic = %F.type (constants.%F.type.0daaa1.1)]
// CHECK:STDOUT:   %F: @F.3.%F.type (%F.type.0daaa1.1) = struct_value () [symbolic = %F (constants.%F.49c1ac.1)]
// CHECK:STDOUT:   %F.specific_fn.loc8_17.2: <specific function> = specific_function %F, @F.2(%Y) [symbolic = %F.specific_fn.loc8_17.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.13320f.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %C.val: @F.3.%C (%C.13320f.2) = struct_value () [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: %empty_struct_type) [thunk @impl.%F.decl.loc8_17.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8: @F.3.%F.type (%F.type.0daaa1.1) = specific_constant @impl.%F.decl.loc8_17.1, @impl(constants.%Y) [symbolic = %F (constants.%F.49c1ac.1)]
// CHECK:STDOUT:     %F.ref: @F.3.%F.type (%F.type.0daaa1.1) = name_ref F, %.loc8 [symbolic = %F (constants.%F.49c1ac.1)]
// CHECK:STDOUT:     %x.ref: %empty_struct_type = name_ref x, %x.param
// CHECK:STDOUT:     %F.specific_fn.loc8_17.1: <specific function> = specific_function %F.ref, @F.2(constants.%Y) [symbolic = %F.specific_fn.loc8_17.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %.1: ref @F.3.%C (%C.13320f.2) = temporary_storage
// CHECK:STDOUT:     %.2: init @F.3.%C (%C.13320f.2) = class_init (), %.1 [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %.3: ref @F.3.%C (%C.13320f.2) = temporary %.1, %.2
// CHECK:STDOUT:     %.4: ref @F.3.%C (%C.13320f.2) = converted %x.ref, %.3
// CHECK:STDOUT:     %.5: @F.3.%C (%C.13320f.2) = bind_value %.4
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc8_17.1(%.5)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.loc5_9.2 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y) {
// CHECK:STDOUT:   %X.loc5_9.2 => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%Y) {
// CHECK:STDOUT:   %Y.loc7_14.2 => constants.%Y
// CHECK:STDOUT:   %C.loc7_25.2 => constants.%C.13320f.2
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type.loc8_17.1 => constants.%F.type.0daaa1.1
// CHECK:STDOUT:   %F.loc8_17.1 => constants.%F.49c1ac.1
// CHECK:STDOUT:   %F.type.loc8_17.2 => constants.%F.type.0daaa1.2
// CHECK:STDOUT:   %F.loc8_17.2 => constants.%F.49c1ac.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C.loc8_14.1 => constants.%C.13320f.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ccc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.3(constants.%Y) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: %empty_tuple.type = bind_symbolic_name X, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.607: type = class_type @C, @C(%empty_tuple) [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.3f3: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.e03 [concrete]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %Y: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic]
// CHECK:STDOUT:   %C.13320f.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.7d9: <witness> = impl_witness imports.%I.impl_witness_table, @impl.22b(%Y) [symbolic]
// CHECK:STDOUT:   %F.type.0daaa1.1: type = fn_type @F.2, @impl.22b(%Y) [symbolic]
// CHECK:STDOUT:   %F.49c1ac.1: %F.type.0daaa1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.ccc: type = pattern_type %C.13320f.2 [symbolic]
// CHECK:STDOUT:   %F.type.0daaa1.2: type = fn_type @F.3, @impl.22b(%Y) [symbolic]
// CHECK:STDOUT:   %F.49c1ac.2: %F.type.0daaa1.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.a35: <witness> = require_complete_type %C.13320f.2 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.8d9: <specific function> = specific_function %F.49c1ac.1, @F.2(%Y) [symbolic]
// CHECK:STDOUT:   %C.val.56a: %C.13320f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.impl_witness.02b: <witness> = impl_witness imports.%I.impl_witness_table, @impl.22b(%empty_tuple) [concrete]
// CHECK:STDOUT:   %F.type.af4856.1: type = fn_type @F.2, @impl.22b(%empty_tuple) [concrete]
// CHECK:STDOUT:   %F.5fa954.1: %F.type.af4856.1 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.af4856.2: type = fn_type @F.3, @impl.22b(%empty_tuple) [concrete]
// CHECK:STDOUT:   %F.5fa954.2: %F.type.af4856.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.607, (%I.impl_witness.02b) [concrete]
// CHECK:STDOUT:   %.885: type = fn_type_with_self_type %F.type.cf0, %I.facet [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.4832e8.1: <specific function> = specific_function %F.5fa954.2, @F.3(%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.186: type = pattern_type %C.607 [concrete]
// CHECK:STDOUT:   %F.specific_fn.4832e8.2: <specific function> = specific_function %F.5fa954.1, @F.2(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.val.12f: %C.607 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl.49c(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.94b: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%C.607) [concrete]
// CHECK:STDOUT:   %Op.type.961: type = fn_type @Op.2, @impl.49c(%C.607) [concrete]
// CHECK:STDOUT:   %Op.7ef: %Op.type.961 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.607, (%Destroy.impl_witness.94b) [concrete]
// CHECK:STDOUT:   %.bb3: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.7ef, @Op.2(%C.607) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//b, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.eb1c17.1: %empty_tuple.type = import_ref Main//b, loc5_9, loaded [symbolic = @C.%X (constants.%X)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//b, loc5_18, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.572 = import_ref Main//b, inst31 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//a, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c44: %I.assoc_type = import_ref Main//a, loc5_14, loaded [concrete = constants.%assoc0.3f3]
// CHECK:STDOUT:   %Main.F.8b9 = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.e03: %F.type.cf0 = import_ref Main//a, loc5_14, loaded [concrete = constants.%F.bc6]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//a, inst19 [no loc], loaded [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %Main.import_ref.f89: <witness> = import_ref Main//b, loc7_32, loaded [symbolic = @impl.22b.%I.impl_witness (constants.%I.impl_witness.7d9)]
// CHECK:STDOUT:   %Main.import_ref.eb1c17.2: %empty_tuple.type = import_ref Main//b, loc7_14, loaded [symbolic = @impl.22b.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.import_ref.dbf: type = import_ref Main//b, loc7_25, loaded [symbolic = @impl.22b.%C (constants.%C.13320f.2)]
// CHECK:STDOUT:   %Main.import_ref.f50: type = import_ref Main//b, loc7_30, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.047: @impl.22b.%F.type.2 (%F.type.0daaa1.2) = import_ref Main//b, loc8_17, loaded [symbolic = @impl.22b.%F.2 (constants.%F.49c1ac.2)]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.047), @impl.22b [concrete]
// CHECK:STDOUT:   %Main.import_ref.eb1c17.3: %empty_tuple.type = import_ref Main//b, loc7_14, loaded [symbolic = @impl.22b.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.import_ref.eb1c17.4: %empty_tuple.type = import_ref Main//b, loc7_14, loaded [symbolic = @impl.22b.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.F.5a8: @impl.22b.%F.type.1 (%F.type.0daaa1.1) = import_ref Main//b, F, loaded [symbolic = @impl.22b.%F.1 (constants.%F.49c1ac.1)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_23, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%Main.import_ref.c44
// CHECK:STDOUT:   witness = (imports.%Main.F.8b9)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.22b(imports.%Main.import_ref.eb1c17.2: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.13320f.2)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @impl.22b(%Y) [symbolic = %I.impl_witness (constants.%I.impl_witness.7d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.2, @impl.22b(%Y) [symbolic = %F.type.1 (constants.%F.type.0daaa1.1)]
// CHECK:STDOUT:   %F.1: @impl.22b.%F.type.1 (%F.type.0daaa1.1) = struct_value () [symbolic = %F.1 (constants.%F.49c1ac.1)]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.3, @impl.22b(%Y) [symbolic = %F.type.2 (constants.%F.type.0daaa1.2)]
// CHECK:STDOUT:   %F.2: @impl.22b.%F.type.2 (%F.type.0daaa1.2) = struct_value () [symbolic = %F.2 (constants.%F.49c1ac.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.dbf as imports.%Main.import_ref.f50 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.f89
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.eb1c17.1: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %X: %empty_tuple.type = bind_symbolic_name X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.572
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc7_6: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_7: %empty_tuple.type = converted %.loc7_6, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_tuple) [concrete = constants.%C.607]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref.loc7_11: %I.assoc_type = name_ref F, imports.%Main.import_ref.c44 [concrete = constants.%assoc0.3f3]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value constants.%C.607, (constants.%I.impl_witness.02b) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_8: %I.type = converted %C, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %impl.elem0.loc7_8: %.885 = impl_witness_access constants.%I.impl_witness.02b, element0 [concrete = constants.%F.5fa954.2]
// CHECK:STDOUT:   %.loc7_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %empty_struct.loc7_16.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17.1: %empty_struct_type = converted %.loc7_16.1, %empty_struct.loc7_16.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn.loc7_8: <specific function> = specific_function %impl.elem0.loc7_8, @F.3(constants.%empty_tuple) [concrete = constants.%F.specific_fn.4832e8.1]
// CHECK:STDOUT:   %empty_struct.loc7_16.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_16.2: %empty_struct_type = converted %.loc7_16.1, %empty_struct.loc7_16.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17.2: %F.type.af4856.1 = specific_constant imports.%Main.F.5a8, @impl.22b(constants.%empty_tuple) [concrete = constants.%F.5fa954.1]
// CHECK:STDOUT:   %F.ref.loc7_17: %F.type.af4856.1 = name_ref F, %.loc7_17.2 [concrete = constants.%F.5fa954.1]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref.loc7_17, @F.2(constants.%empty_tuple) [concrete = constants.%F.specific_fn.4832e8.2]
// CHECK:STDOUT:   %.loc7_16.3: ref %C.607 = temporary_storage
// CHECK:STDOUT:   %.loc7_16.4: init %C.607 = class_init (), %.loc7_16.3 [concrete = constants.%C.val.12f]
// CHECK:STDOUT:   %.loc7_16.5: ref %C.607 = temporary %.loc7_16.3, %.loc7_16.4
// CHECK:STDOUT:   %.loc7_16.6: ref %C.607 = converted %.loc7_16.2, %.loc7_16.5
// CHECK:STDOUT:   %.loc7_16.7: %C.607 = bind_value %.loc7_16.6
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc7_16.7)
// CHECK:STDOUT:   %impl.elem0.loc7_16: %.bb3 = impl_witness_access constants.%Destroy.impl_witness.94b, element0 [concrete = constants.%Op.7ef]
// CHECK:STDOUT:   %bound_method.loc7_16.1: <bound method> = bound_method %.loc7_16.3, %impl.elem0.loc7_16
// CHECK:STDOUT:   %specific_fn.loc7_16: <specific function> = specific_function %impl.elem0.loc7_16, @Op.2(constants.%C.607) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_16.2: <bound method> = bound_method %.loc7_16.3, %specific_fn.loc7_16
// CHECK:STDOUT:   %.loc7_16.8: %C.607 = bind_value %.loc7_16.3
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc7_16.2(%.loc7_16.8)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5dd: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(imports.%Main.import_ref.eb1c17.3: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.13320f.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.ccc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete.a35)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.3(imports.%Main.import_ref.eb1c17.4: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = bind_symbolic_name Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl.22b(%Y) [symbolic = %F.type (constants.%F.type.0daaa1.1)]
// CHECK:STDOUT:   %F: @F.3.%F.type (%F.type.0daaa1.1) = struct_value () [symbolic = %F (constants.%F.49c1ac.1)]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F.2(%Y) [symbolic = %F.specific_fn (constants.%F.specific_fn.8d9)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.13320f.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete.a35)]
// CHECK:STDOUT:   %C.val: @F.3.%C (%C.13320f.2) = struct_value () [symbolic = %C.val (constants.%C.val.56a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn [thunk imports.%Main.F.5a8];
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_tuple) {
// CHECK:STDOUT:   %X => constants.%empty_tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y) {
// CHECK:STDOUT:   %X => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.22b(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.13320f.2
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.7d9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type.1 => constants.%F.type.0daaa1.1
// CHECK:STDOUT:   %F.1 => constants.%F.49c1ac.1
// CHECK:STDOUT:   %F.type.2 => constants.%F.type.0daaa1.2
// CHECK:STDOUT:   %F.2 => constants.%F.49c1ac.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.13320f.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ccc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.3(constants.%Y) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.22b(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.607
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.02b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type.1 => constants.%F.type.af4856.1
// CHECK:STDOUT:   %F.1 => constants.%F.5fa954.1
// CHECK:STDOUT:   %F.type.2 => constants.%F.type.af4856.2
// CHECK:STDOUT:   %F.2 => constants.%F.5fa954.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.3(constants.%empty_tuple) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %F.type => constants.%F.type.af4856.1
// CHECK:STDOUT:   %F => constants.%F.5fa954.1
// CHECK:STDOUT:   %F.specific_fn => constants.%F.specific_fn.4832e8.2
// CHECK:STDOUT:   %C => constants.%C.607
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %C.val => constants.%C.val.12f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.607
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.186
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
