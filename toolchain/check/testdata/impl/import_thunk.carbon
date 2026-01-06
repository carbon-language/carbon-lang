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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc5_12.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%I.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn(%x.param: %empty_struct_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %X: %empty_tuple.type = symbolic_binding X, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.c8540f.1: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic]
// CHECK:STDOUT:   %C.c8540f.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.651: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.fb2: type = pattern_type %C.c8540f.2 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.cf838e.1: type = fn_type @C.as.I.impl.F.loc8_17.1, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.421e41.1: %C.as.I.impl.F.type.cf838e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.c8540f.2, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.cf838e.2: type = fn_type @C.as.I.impl.F.loc8_17.2, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.421e41.2: %C.as.I.impl.F.type.cf838e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.c8540f.2 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F.421e41.1, @C.as.I.impl.F.loc8_17.1(%Y) [symbolic]
// CHECK:STDOUT:   %C.val: %C.c8540f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.971 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.809: <witness> = custom_witness (%DestroyOp), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.17e: %Destroy.type = facet_value %C.c8540f.2, (%custom_witness.809) [symbolic]
// CHECK:STDOUT:   %.3ae: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.17e [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.c82 = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.ef2 = import_ref Main//a, loc5_14, unloaded
// CHECK:STDOUT:   %Main.F: %I.F.type = import_ref Main//a, F, loaded [concrete = constants.%I.F]
// CHECK:STDOUT:   %Main.import_ref.236: %I.type = import_ref Main//a, loc4_13, loaded [symbolic = constants.%Self.651]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.446: %Destroy.assoc_type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.import_ref.971: %Destroy.Op.type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%Destroy.Op]
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
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_14.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc5_14.3: type = converted %.loc5_14.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc5_9.2: %empty_tuple.type = symbolic_binding X, 0 [symbolic = %X.loc5_9.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Y.ref: %empty_tuple.type = name_ref Y, %Y.loc7_14.2 [symbolic = %Y.loc7_14.1 (constants.%Y)]
// CHECK:STDOUT:     %C.loc7_25.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc7_25.1 (constants.%C.c8540f.2)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_19.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc7_19.3: type = converted %.loc7_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.loc7_14.2: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y.loc7_14.1 (constants.%Y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c82
// CHECK:STDOUT:   .F = imports.%Main.import_ref.ef2
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%Y.loc7_14.2: %empty_tuple.type) {
// CHECK:STDOUT:   %Y.loc7_14.1: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y.loc7_14.1 (constants.%Y)]
// CHECK:STDOUT:   %C.loc7_25.1: type = class_type @C, @C(%Y.loc7_14.1) [symbolic = %C.loc7_25.1 (constants.%C.c8540f.2)]
// CHECK:STDOUT:   %I.impl_witness.loc7_32.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%Y.loc7_14.1) [symbolic = %I.impl_witness.loc7_32.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_17.1: type = fn_type @C.as.I.impl.F.loc8_17.1, @C.as.I.impl(%Y.loc7_14.1) [symbolic = %C.as.I.impl.F.type.loc8_17.1 (constants.%C.as.I.impl.F.type.cf838e.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_17.1: @C.as.I.impl.%C.as.I.impl.F.type.loc8_17.1 (%C.as.I.impl.F.type.cf838e.1) = struct_value () [symbolic = %C.as.I.impl.F.loc8_17.1 (constants.%C.as.I.impl.F.421e41.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_17.2: type = fn_type @C.as.I.impl.F.loc8_17.2, @C.as.I.impl(%Y.loc7_14.1) [symbolic = %C.as.I.impl.F.type.loc8_17.2 (constants.%C.as.I.impl.F.type.cf838e.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_17.2: @C.as.I.impl.%C.as.I.impl.F.type.loc8_17.2 (%C.as.I.impl.F.type.cf838e.2) = struct_value () [symbolic = %C.as.I.impl.F.loc8_17.2 (constants.%C.as.I.impl.F.421e41.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_25.2 as %I.ref {
// CHECK:STDOUT:     %C.as.I.impl.F.decl.loc8_17.1: @C.as.I.impl.%C.as.I.impl.F.type.loc8_17.1 (%C.as.I.impl.F.type.cf838e.1) = fn_decl @C.as.I.impl.F.loc8_17.1 [symbolic = @C.as.I.impl.%C.as.I.impl.F.loc8_17.1 (constants.%C.as.I.impl.F.421e41.1)] {
// CHECK:STDOUT:       %x.patt: @C.as.I.impl.F.loc8_17.1.%pattern_type (%pattern_type.fb2) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @C.as.I.impl.F.loc8_17.1.%pattern_type (%pattern_type.fb2) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @C.as.I.impl.F.loc8_17.1.%C.loc8_14.1 (%C.c8540f.2) = value_param call_param0
// CHECK:STDOUT:       %.loc8: type = splice_block %C.loc8_14.2 [symbolic = %C.loc8_14.1 (constants.%C.c8540f.2)] {
// CHECK:STDOUT:         %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:         %Y.ref: %empty_tuple.type = name_ref Y, @C.as.I.impl.%Y.loc7_14.2 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:         %C.loc8_14.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc8_14.1 (constants.%C.c8540f.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @C.as.I.impl.F.loc8_17.1.%C.loc8_14.1 (%C.c8540f.2) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.as.I.impl.F.decl.loc8_17.2: @C.as.I.impl.%C.as.I.impl.F.type.loc8_17.2 (%C.as.I.impl.F.type.cf838e.2) = fn_decl @C.as.I.impl.F.loc8_17.2 [symbolic = @C.as.I.impl.%C.as.I.impl.F.loc8_17.2 (constants.%C.as.I.impl.F.421e41.2)] {
// CHECK:STDOUT:       %x.patt: %pattern_type.a96 = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: %pattern_type.a96 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:       %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl.loc8_17.2), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_32.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%Y) [symbolic = %I.impl_witness.loc7_32.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .Y = <poisoned>
// CHECK:STDOUT:     .F = %C.as.I.impl.F.decl.loc8_17.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_32.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%X.loc5_9.2: %empty_tuple.type) {
// CHECK:STDOUT:   %X.loc5_9.1: %empty_tuple.type = symbolic_binding X, 0 [symbolic = %X.loc5_9.1 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.c8540f.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.236: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.loc8_17.1(@C.as.I.impl.%Y.loc7_14.2: %empty_tuple.type) {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.loc8_14.1: type = class_type @C, @C(%Y) [symbolic = %C.loc8_14.1 (constants.%C.c8540f.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc8_14.1 [symbolic = %pattern_type (constants.%pattern_type.fb2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc8_14.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @C.as.I.impl.F.loc8_17.1.%C.loc8_14.1 (%C.c8540f.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.loc8_17.2(@C.as.I.impl.%Y.loc7_14.2: %empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F.loc8_17.1, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.cf838e.1)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.F.loc8_17.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.cf838e.1) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.421e41.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.loc8_17.2: <specific function> = specific_function %C.as.I.impl.F, @C.as.I.impl.F.loc8_17.1(%Y) [symbolic = %C.as.I.impl.F.specific_fn.loc8_17.2 (constants.%C.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.c8540f.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %C.val: @C.as.I.impl.F.loc8_17.2.%C (%C.c8540f.2) = struct_value () [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.fb2)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (constants.%custom_witness.809) [symbolic = %Destroy.facet (constants.%Destroy.facet.17e)]
// CHECK:STDOUT:   %.6: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.6 (constants.%.3ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: %empty_struct_type) [thunk @C.as.I.impl.%C.as.I.impl.F.decl.loc8_17.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8: @C.as.I.impl.F.loc8_17.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.cf838e.1) = specific_constant @C.as.I.impl.%C.as.I.impl.F.decl.loc8_17.1, @C.as.I.impl(constants.%Y) [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.421e41.1)]
// CHECK:STDOUT:     %F.ref: @C.as.I.impl.F.loc8_17.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.cf838e.1) = name_ref F, %.loc8 [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.421e41.1)]
// CHECK:STDOUT:     %x.ref: %empty_struct_type = name_ref x, %x.param
// CHECK:STDOUT:     %C.as.I.impl.F.specific_fn.loc8_17.1: <specific function> = specific_function %F.ref, @C.as.I.impl.F.loc8_17.1(constants.%Y) [symbolic = %C.as.I.impl.F.specific_fn.loc8_17.2 (constants.%C.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %.1: ref @C.as.I.impl.F.loc8_17.2.%C (%C.c8540f.2) = temporary_storage
// CHECK:STDOUT:     %.2: init @C.as.I.impl.F.loc8_17.2.%C (%C.c8540f.2) = class_init (), %.1 [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %.3: ref @C.as.I.impl.F.loc8_17.2.%C (%C.c8540f.2) = temporary %.1, %.2
// CHECK:STDOUT:     %.4: ref @C.as.I.impl.F.loc8_17.2.%C (%C.c8540f.2) = converted %x.ref, %.3
// CHECK:STDOUT:     %.5: @C.as.I.impl.F.loc8_17.2.%C (%C.c8540f.2) = acquire_value %.4
// CHECK:STDOUT:     %C.as.I.impl.F.call: init %empty_tuple.type = call %C.as.I.impl.F.specific_fn.loc8_17.1(%.5)
// CHECK:STDOUT:     %Op.ref: %Destroy.assoc_type = name_ref Op, imports.%Core.import_ref.446 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0: @C.as.I.impl.F.loc8_17.2.%.6 (%.3ae) = impl_witness_access constants.%custom_witness.809, element0 [concrete = constants.%DestroyOp]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %.3, %impl.elem0
// CHECK:STDOUT:     %DestroyOp.call: init %empty_tuple.type = call %bound_method(%.3)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: @C.as.I.impl.F.loc8_17.2.%C (%C.c8540f.2)) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.loc5_9.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y) {
// CHECK:STDOUT:   %X.loc5_9.1 => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Y) {
// CHECK:STDOUT:   %Y.loc7_14.1 => constants.%Y
// CHECK:STDOUT:   %C.loc7_25.1 => constants.%C.c8540f.2
// CHECK:STDOUT:   %I.impl_witness.loc7_32.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_17.1 => constants.%C.as.I.impl.F.type.cf838e.1
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_17.1 => constants.%C.as.I.impl.F.421e41.1
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_17.2 => constants.%C.as.I.impl.F.type.cf838e.2
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_17.2 => constants.%C.as.I.impl.F.421e41.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.651) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.loc8_17.1(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C.loc8_14.1 => constants.%C.c8540f.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fb2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.loc8_17.2(constants.%Y) {}
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
// CHECK:STDOUT:   %X: %empty_tuple.type = symbolic_binding X, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.387: type = class_type @C, @C(%empty_tuple) [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.651: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.a72: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.b58 [concrete]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic]
// CHECK:STDOUT:   %C.c8540f.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.a8f: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.cf838e.1: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.421e41.1: %C.as.I.impl.F.type.cf838e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.1 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.cf838e.2: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.421e41.2: %C.as.I.impl.F.type.cf838e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.fb2: type = pattern_type %C.c8540f.2 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.c8540f.2 [symbolic]
// CHECK:STDOUT:   %custom_witness.854d64.1: <witness> = custom_witness (%DestroyOp.b0ebf8.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.23d: %Destroy.type = facet_value %C.c8540f.2, (%custom_witness.854d64.1) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %.7aa: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.23d [symbolic]
// CHECK:STDOUT:   %C.val.6cb: %C.c8540f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.23d: <specific function> = specific_function %C.as.I.impl.F.421e41.2, @C.as.I.impl.F.2(%Y) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.ead: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.0e1915.1: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.fcfec4.1: %C.as.I.impl.F.type.0e1915.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.0e1915.2: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.fcfec4.2: %C.as.I.impl.F.type.0e1915.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.387, (%I.impl_witness.ead) [concrete]
// CHECK:STDOUT:   %.243: type = fn_type_with_self_type %I.F.type, %I.facet [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.a77113.1: <specific function> = specific_function %C.as.I.impl.F.fcfec4.2, @C.as.I.impl.F.1(%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.678: type = pattern_type %C.387 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.a77113.2: <specific function> = specific_function %C.as.I.impl.F.fcfec4.1, @C.as.I.impl.F.2(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.val.135: %C.387 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc7 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d0a3ec.2: %Destroy.type = facet_value %C.387, (%custom_witness.854d64.1) [concrete]
// CHECK:STDOUT:   %.33898c.2: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.d0a3ec.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//b, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//b, loc5_18, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.dbf = import_ref Main//b, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c90e1a.1: %empty_tuple.type = import_ref Main//b, loc5_9, loaded [symbolic = @C.%X (constants.%X)]
// CHECK:STDOUT:   %Main.import_ref.c82 = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.ce7: %I.assoc_type = import_ref Main//a, loc5_14, loaded [concrete = constants.%assoc0.a72]
// CHECK:STDOUT:   %Main.F.ff3 = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.b58: %I.F.type = import_ref Main//a, loc5_14, loaded [concrete = constants.%I.F]
// CHECK:STDOUT:   %Main.import_ref.236: %I.type = import_ref Main//a, loc4_13, loaded [symbolic = constants.%Self.651]
// CHECK:STDOUT:   %Main.import_ref.e97: <witness> = import_ref Main//b, loc7_32, loaded [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness.a8f)]
// CHECK:STDOUT:   %Main.import_ref.045: @C.as.I.impl.%C.as.I.impl.F.type.2 (%C.as.I.impl.F.type.cf838e.1) = import_ref Main//b, loc8_17, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F.2 (constants.%C.as.I.impl.F.421e41.1)]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.045), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.F.13b: @C.as.I.impl.%C.as.I.impl.F.type.1 (%C.as.I.impl.F.type.cf838e.2) = import_ref Main//b, F, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F.1 (constants.%C.as.I.impl.F.421e41.2)]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.2: %empty_tuple.type = import_ref Main//b, loc7_14, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.3: %empty_tuple.type = import_ref Main//b, loc7_14, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.import_ref.2e2: type = import_ref Main//b, loc7_25, loaded [symbolic = @C.as.I.impl.%C (constants.%C.c8540f.2)]
// CHECK:STDOUT:   %Main.import_ref.91f: type = import_ref Main//b, loc7_30, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.4: %empty_tuple.type = import_ref Main//b, loc7_14, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c82
// CHECK:STDOUT:   .F = imports.%Main.import_ref.ce7
// CHECK:STDOUT:   witness = (imports.%Main.F.ff3)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(imports.%Main.import_ref.c90e1a.4: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.c8540f.2)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic = %I.impl_witness (constants.%I.impl_witness.a8f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type.1 (constants.%C.as.I.impl.F.type.cf838e.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.1: @C.as.I.impl.%C.as.I.impl.F.type.1 (%C.as.I.impl.F.type.cf838e.2) = struct_value () [symbolic = %C.as.I.impl.F.1 (constants.%C.as.I.impl.F.421e41.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.type.2: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type.2 (constants.%C.as.I.impl.F.type.cf838e.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.2: @C.as.I.impl.%C.as.I.impl.F.type.2 (%C.as.I.impl.F.type.cf838e.1) = struct_value () [symbolic = %C.as.I.impl.F.2 (constants.%C.as.I.impl.F.421e41.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.2e2 as imports.%Main.import_ref.91f {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.e97
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.c90e1a.1: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %X: %empty_tuple.type = symbolic_binding X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.dbf
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc7_6: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_7: %empty_tuple.type = converted %.loc7_6, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_tuple) [concrete = constants.%C.387]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref.loc7_11: %I.assoc_type = name_ref F, imports.%Main.import_ref.ce7 [concrete = constants.%assoc0.a72]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (constants.%I.impl_witness.ead) [concrete = constants.%I.facet]
// CHECK:STDOUT:   %.loc7_8: %I.type = converted %C, %I.facet [concrete = constants.%I.facet]
// CHECK:STDOUT:   %impl.elem0: %.243 = impl_witness_access constants.%I.impl_witness.ead, element0 [concrete = constants.%C.as.I.impl.F.fcfec4.2]
// CHECK:STDOUT:   %.loc7_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %empty_struct.loc7_16.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17.1: %empty_struct_type = converted %.loc7_16.1, %empty_struct.loc7_16.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.I.impl.F.1(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.specific_fn.a77113.1]
// CHECK:STDOUT:   %empty_struct.loc7_16.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_16.2: %empty_struct_type = converted %.loc7_16.1, %empty_struct.loc7_16.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17.2: %C.as.I.impl.F.type.0e1915.1 = specific_constant imports.%Main.F.13b, @C.as.I.impl(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.fcfec4.1]
// CHECK:STDOUT:   %F.ref.loc7_17: %C.as.I.impl.F.type.0e1915.1 = name_ref F, %.loc7_17.2 [concrete = constants.%C.as.I.impl.F.fcfec4.1]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %F.ref.loc7_17, @C.as.I.impl.F.2(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.specific_fn.a77113.2]
// CHECK:STDOUT:   %.loc7_16.3: ref %C.387 = temporary_storage
// CHECK:STDOUT:   %.loc7_16.4: init %C.387 = class_init (), %.loc7_16.3 [concrete = constants.%C.val.135]
// CHECK:STDOUT:   %.loc7_16.5: ref %C.387 = temporary %.loc7_16.3, %.loc7_16.4
// CHECK:STDOUT:   %.loc7_16.6: ref %C.387 = converted %.loc7_16.2, %.loc7_16.5
// CHECK:STDOUT:   %.loc7_16.7: %C.387 = acquire_value %.loc7_16.6
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %C.as.I.impl.F.specific_fn(%.loc7_16.7)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc7_16.5, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc7_16.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.236: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.1(imports.%Main.import_ref.c90e1a.2: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.cf838e.2)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.F.1.%C.as.I.impl.F.type (%C.as.I.impl.F.type.cf838e.2) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.421e41.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F, @C.as.I.impl.F.2(%Y) [symbolic = %C.as.I.impl.F.specific_fn (constants.%C.as.I.impl.F.specific_fn.23d)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.c8540f.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %C.val: @C.as.I.impl.F.1.%C (%C.c8540f.2) = struct_value () [symbolic = %C.val (constants.%C.val.6cb)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.fb2)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (constants.%custom_witness.854d64.1) [symbolic = %Destroy.facet (constants.%Destroy.facet.23d)]
// CHECK:STDOUT:   %.1: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.1 (constants.%.7aa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn [thunk imports.%Main.F.13b];
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.1 = "no_op" [from "b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.2(imports.%Main.import_ref.c90e1a.3: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.c8540f.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.fb2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc7(%self.param: %C.387) = "no_op";
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
// CHECK:STDOUT: specific @I.F(constants.%Self.651) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y) {
// CHECK:STDOUT:   %X => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.c8540f.2
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.a8f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1 => constants.%C.as.I.impl.F.type.cf838e.2
// CHECK:STDOUT:   %C.as.I.impl.F.1 => constants.%C.as.I.impl.F.421e41.2
// CHECK:STDOUT:   %C.as.I.impl.F.type.2 => constants.%C.as.I.impl.F.type.cf838e.1
// CHECK:STDOUT:   %C.as.I.impl.F.2 => constants.%C.as.I.impl.F.421e41.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.2(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.c8540f.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fb2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.1(constants.%Y) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.387
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.ead
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1 => constants.%C.as.I.impl.F.type.0e1915.1
// CHECK:STDOUT:   %C.as.I.impl.F.1 => constants.%C.as.I.impl.F.fcfec4.1
// CHECK:STDOUT:   %C.as.I.impl.F.type.2 => constants.%C.as.I.impl.F.type.0e1915.2
// CHECK:STDOUT:   %C.as.I.impl.F.2 => constants.%C.as.I.impl.F.fcfec4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.1(constants.%empty_tuple) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C.as.I.impl.F.type => constants.%C.as.I.impl.F.type.0e1915.1
// CHECK:STDOUT:   %C.as.I.impl.F => constants.%C.as.I.impl.F.fcfec4.1
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn => constants.%C.as.I.impl.F.specific_fn.a77113.2
// CHECK:STDOUT:   %C => constants.%C.387
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %C.val => constants.%C.val.135
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.678
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.d0a3ec.2
// CHECK:STDOUT:   %.1 => constants.%.33898c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.2(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.387
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.678
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
