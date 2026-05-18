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
  fn F(unused x: C(Y)) {}
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
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F: %I.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
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
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F)] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc5_12.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn(%x.param: %empty_struct_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %X: %empty_tuple.type = symbolic_binding X, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.9015c3.1: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic]
// CHECK:STDOUT:   %C.9015c3.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.c30: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.de0: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.c30) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.8e2: %I.WithSelf.F.type.de0 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %pattern_type.556: type = pattern_type %C.9015c3.2 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.a3320c.1: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.6c5a1c.1: %C.as.I.impl.F.type.a3320c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.9015c3.2, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.8eb: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.81d: %I.WithSelf.F.type.8eb = struct_value () [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.a3320c.2: type = fn_type @C.as.I.impl.F.loc8_24.2, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.6c5a1c.2: %C.as.I.impl.F.type.a3320c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.9015c3.2 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F.6c5a1c.1, @C.as.I.impl.F.loc8_24.1(%Y) [symbolic]
// CHECK:STDOUT:   %C.val: %C.9015c3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.af3: ref %C.9015c3.2 = temporary invalid, %C.val [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.761: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b9d: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.761) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.481: %Destroy.WithSelf.Op.type.b9d = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.8cc [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.9015c3.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.33b: %Destroy.type = facet_value %C.9015c3.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b7e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.33b) [symbolic]
// CHECK:STDOUT:   %.7e5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b7e, %Destroy.facet.33b [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.7e5 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %bound_method.c0c: <bound method> = bound_method %.af3, %impl.elem0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.WithSelf.Op(%Destroy.facet.33b) [symbolic]
// CHECK:STDOUT:   %bound_method.e8f: <bound method> = bound_method %.af3, %specific_impl_fn [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.da4 = import_ref Main//a, loc5_14, unloaded
// CHECK:STDOUT:   %Main.F: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.de0) = import_ref Main//a, F, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.8e2)]
// CHECK:STDOUT:   %I.WithSelf.F.decl: %I.WithSelf.F.type.de0 = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.8e2)] {} {}
// CHECK:STDOUT:   %Main.import_ref.f4a753.2: %I.type = import_ref Main//a, loc4_13, loaded [symbolic = constants.%Self.c30]
// CHECK:STDOUT:   %Main.import_ref.366 = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.b34: %Destroy.assoc_type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.import_ref.8cc: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.b9d) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.481)]
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
// CHECK:STDOUT:     %X.loc5_10.2: %empty_tuple.type = symbolic_binding X, 0 [symbolic = %X.loc5_10.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Y.ref: %empty_tuple.type = name_ref Y, %Y.loc7_15.2 [symbolic = %Y.loc7_15.1 (constants.%Y)]
// CHECK:STDOUT:     %C.loc7_25.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc7_25.1 (constants.%C.9015c3.2)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_19.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc7_19.3: type = converted %.loc7_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.loc7_15.2: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y.loc7_15.1 (constants.%Y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.366
// CHECK:STDOUT:   .F = imports.%Main.import_ref.da4
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%Y.loc7_15.2: %empty_tuple.type) {
// CHECK:STDOUT:   %Y.loc7_15.1: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y.loc7_15.1 (constants.%Y)]
// CHECK:STDOUT:   %C.loc7_25.1: type = class_type @C, @C(%Y.loc7_15.1) [symbolic = %C.loc7_25.1 (constants.%C.9015c3.2)]
// CHECK:STDOUT:   %I.impl_witness.loc7_32.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %I.impl_witness.loc7_32.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.1: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %C.as.I.impl.F.type.loc8_24.1 (constants.%C.as.I.impl.F.type.a3320c.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.1: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.1 (%C.as.I.impl.F.type.a3320c.1) = struct_value () [symbolic = %C.as.I.impl.F.loc8_24.1 (constants.%C.as.I.impl.F.6c5a1c.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.2: type = fn_type @C.as.I.impl.F.loc8_24.2, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %C.as.I.impl.F.type.loc8_24.2 (constants.%C.as.I.impl.F.type.a3320c.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.2: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.2 (%C.as.I.impl.F.type.a3320c.2) = struct_value () [symbolic = %C.as.I.impl.F.loc8_24.2 (constants.%C.as.I.impl.F.6c5a1c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_25.2 as %I.ref {
// CHECK:STDOUT:     %C.as.I.impl.F.decl.loc8_24.1: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.1 (%C.as.I.impl.F.type.a3320c.1) = fn_decl @C.as.I.impl.F.loc8_24.1 [symbolic = @C.as.I.impl.%C.as.I.impl.F.loc8_24.1 (constants.%C.as.I.impl.F.6c5a1c.1)] {
// CHECK:STDOUT:       %x.param_patt: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.556) = value_param_pattern [concrete]
// CHECK:STDOUT:       %x.patt: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.556) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.9015c3.2) = value_param call_param0
// CHECK:STDOUT:       %.loc8: type = splice_block %C.loc8_21.2 [symbolic = %C.loc8_21.1 (constants.%C.9015c3.2)] {
// CHECK:STDOUT:         %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:         %Y.ref: %empty_tuple.type = name_ref Y, @C.as.I.impl.%Y.loc7_15.2 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:         %C.loc8_21.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc8_21.1 (constants.%C.9015c3.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.9015c3.2) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.as.I.impl.F.decl.loc8_24.2: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.2 (%C.as.I.impl.F.type.a3320c.2) = fn_decl @C.as.I.impl.F.loc8_24.2 [symbolic = @C.as.I.impl.%C.as.I.impl.F.loc8_24.2 (constants.%C.as.I.impl.F.6c5a1c.2)] {
// CHECK:STDOUT:       %x.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:       %x.patt: %pattern_type.a96 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:       %x: %empty_struct_type = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl.loc8_24.2), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_32.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%Y) [symbolic = %I.impl_witness.loc7_32.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .Y = <poisoned>
// CHECK:STDOUT:     .F = %C.as.I.impl.F.decl.loc8_24.1
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_32.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%X.loc5_10.2: %empty_tuple.type) {
// CHECK:STDOUT:   %X.loc5_10.1: %empty_tuple.type = symbolic_binding X, 0 [symbolic = %X.loc5_10.1 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.9015c3.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.f4a753.2: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.loc8_24.1(@C.as.I.impl.%Y.loc7_15.2: %empty_tuple.type) {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.loc8_21.1: type = class_type @C, @C(%Y) [symbolic = %C.loc8_21.1 (constants.%C.9015c3.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc8_21.1 [symbolic = %pattern_type (constants.%pattern_type.556)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc8_21.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.9015c3.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.loc8_24.2(@C.as.I.impl.%Y.loc7_15.2: %empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.a3320c.1)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.a3320c.1) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.6c5a1c.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.loc8_24.2: <specific function> = specific_function %C.as.I.impl.F, @C.as.I.impl.F.loc8_24.1(%Y) [symbolic = %C.as.I.impl.F.specific_fn.loc8_24.2 (constants.%C.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.9015c3.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %C.val: @C.as.I.impl.F.loc8_24.2.%C (%C.9015c3.2) = struct_value () [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:   %.6: ref @C.as.I.impl.F.loc8_24.2.%C (%C.9015c3.2) = temporary invalid, %C.val [symbolic = %.6 (constants.%.af3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.33b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b7e)]
// CHECK:STDOUT:   %.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.7 (constants.%.7e5)]
// CHECK:STDOUT:   %impl.elem0.2: @C.as.I.impl.F.loc8_24.2.%.7 (%.7e5) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %.6, %impl.elem0.2 [symbolic = %bound_method.3 (constants.%bound_method.c0c)]
// CHECK:STDOUT:   %specific_impl_fn.2: <specific function> = specific_impl_function %impl.elem0.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %bound_method.4: <bound method> = bound_method %.6, %specific_impl_fn.2 [symbolic = %bound_method.4 (constants.%bound_method.e8f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: %empty_struct_type) [thunk @C.as.I.impl.%C.as.I.impl.F.decl.loc8_24.1 for imports.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet)] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.a3320c.1) = specific_constant @C.as.I.impl.%C.as.I.impl.F.decl.loc8_24.1, @C.as.I.impl(constants.%Y) [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.6c5a1c.1)]
// CHECK:STDOUT:     %F.ref: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.a3320c.1) = name_ref F, %.loc8 [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.6c5a1c.1)]
// CHECK:STDOUT:     %C.as.I.impl.F.specific_fn.loc8_24.1: <specific function> = specific_function %F.ref, @C.as.I.impl.F.loc8_24.1(constants.%Y) [symbolic = %C.as.I.impl.F.specific_fn.loc8_24.2 (constants.%C.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %.1: ref @C.as.I.impl.F.loc8_24.2.%C (%C.9015c3.2) = temporary_storage
// CHECK:STDOUT:     %.2: init @C.as.I.impl.F.loc8_24.2.%C (%C.9015c3.2) to %.1 = class_init () [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %.3: init @C.as.I.impl.F.loc8_24.2.%C (%C.9015c3.2) = converted %x.param, %.2 [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %.4: ref @C.as.I.impl.F.loc8_24.2.%C (%C.9015c3.2) = temporary %.1, %.3 [symbolic = %.6 (constants.%.af3)]
// CHECK:STDOUT:     %.5: @C.as.I.impl.F.loc8_24.2.%C (%C.9015c3.2) = acquire_value %.4 [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %C.as.I.impl.F.call: init %empty_tuple.type = call %C.as.I.impl.F.specific_fn.loc8_24.1(%.5)
// CHECK:STDOUT:     %Op.ref: %Destroy.assoc_type = name_ref Op, imports.%Core.import_ref.b34 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.1: @C.as.I.impl.F.loc8_24.2.%.7 (%.7e5) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.1: <bound method> = bound_method %.4, %impl.elem0.1 [symbolic = %bound_method.3 (constants.%bound_method.c0c)]
// CHECK:STDOUT:     %specific_impl_fn.1: <specific function> = specific_impl_function %impl.elem0.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.33b) [symbolic = %specific_impl_fn.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.2: <bound method> = bound_method %.4, %specific_impl_fn.1 [symbolic = %bound_method.4 (constants.%bound_method.e8f)]
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.2(%.4)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.loc5_10.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y) {
// CHECK:STDOUT:   %X.loc5_10.1 => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.c30) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c30
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.de0
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.8e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.c30) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Y) {
// CHECK:STDOUT:   %Y.loc7_15.1 => constants.%Y
// CHECK:STDOUT:   %C.loc7_25.1 => constants.%C.9015c3.2
// CHECK:STDOUT:   %I.impl_witness.loc7_32.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.1 => constants.%C.as.I.impl.F.type.a3320c.1
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.1 => constants.%C.as.I.impl.F.6c5a1c.1
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.2 => constants.%C.as.I.impl.F.type.a3320c.2
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.2 => constants.%C.as.I.impl.F.6c5a1c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.loc8_24.1(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C.loc8_21.1 => constants.%C.9015c3.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.556
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.8eb
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.81d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.loc8_24.2(constants.%Y) {}
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
// CHECK:STDOUT:   %C.144: type = class_type @C, @C(%empty_tuple) [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.c30: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.de0: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.c30) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.8e2: %I.WithSelf.F.type.de0 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.c00: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.f77 [concrete]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic]
// CHECK:STDOUT:   %C.9015c3.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.243: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.029cdf.1: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.49b6ba.1: %C.as.I.impl.F.type.029cdf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.029cdf.2: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.49b6ba.2: %C.as.I.impl.F.type.029cdf.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.556: type = pattern_type %C.9015c3.2 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.9015c3.2 [symbolic]
// CHECK:STDOUT:   %I.facet.a37: %I.type = facet_value %C.9015c3.2, (%I.impl_witness.243) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.9015c3.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.33b: %Destroy.type = facet_value %C.9015c3.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b7e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.33b) [symbolic]
// CHECK:STDOUT:   %.7e5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b7e, %Destroy.facet.33b [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.7e5 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.WithSelf.Op(%Destroy.facet.33b) [symbolic]
// CHECK:STDOUT:   %C.val.130: %C.9015c3.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.af3: ref %C.9015c3.2 = temporary invalid, %C.val.130 [symbolic]
// CHECK:STDOUT:   %bound_method.e8f: <bound method> = bound_method %.af3, %specific_impl_fn [symbolic]
// CHECK:STDOUT:   %bound_method.c0c: <bound method> = bound_method %.af3, %impl.elem0 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.219: <specific function> = specific_function %C.as.I.impl.F.49b6ba.2, @C.as.I.impl.F.2(%Y) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.b9d: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.0f4092.1: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.2e70f7.1: %C.as.I.impl.F.type.0f4092.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.0f4092.2: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.2e70f7.2: %C.as.I.impl.F.type.0f4092.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.bae: %I.type = facet_value %C.144, (%I.impl_witness.b9d) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.be4: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.bae) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ea9: %I.WithSelf.F.type.be4 = struct_value () [concrete]
// CHECK:STDOUT:   %.612: type = fn_type_with_self_type %I.WithSelf.F.type.be4, %I.facet.bae [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.157186.1: <specific function> = specific_function %C.as.I.impl.F.2e70f7.2, @C.as.I.impl.F.1(%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.02a: type = pattern_type %C.144 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.157186.2: <specific function> = specific_function %C.as.I.impl.F.2e70f7.1, @C.as.I.impl.F.2(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.val.2c6: %C.144 = struct_value () [concrete]
// CHECK:STDOUT:   %.6f8: ref %C.144 = temporary invalid, %C.val.2c6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.2: type = fn_type @Destroy.Op.loc7_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.2: %Destroy.Op.type.3f5f78.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1367bd.2: <witness> = custom_witness (%Destroy.Op.f0ba5f.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.06b: %Destroy.type = facet_value %C.144, (%custom_witness.1367bd.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.417: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.06b) [concrete]
// CHECK:STDOUT:   %.6e3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.417, %Destroy.facet.06b [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.6f8, %Destroy.Op.f0ba5f.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//b, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Core.a3bffd.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//b, loc5_18, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.602 = import_ref Main//b, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c90e1a.1: %empty_tuple.type = import_ref Main//b, loc5_10, loaded [symbolic = @C.%X (constants.%X)]
// CHECK:STDOUT:   %Main.import_ref.7f7: %I.assoc_type = import_ref Main//a, loc5_14, loaded [concrete = constants.%assoc0.c00]
// CHECK:STDOUT:   %Main.F.09a = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %I.WithSelf.F.decl: %I.WithSelf.F.type.de0 = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.8e2)] {} {}
// CHECK:STDOUT:   %Main.import_ref.f4a753.2: %I.type = import_ref Main//a, loc4_13, loaded [symbolic = constants.%Self.c30]
// CHECK:STDOUT:   %Main.import_ref.366 = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.f77: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.de0) = import_ref Main//a, loc5_14, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.8e2)]
// CHECK:STDOUT:   %Main.import_ref.867: <witness> = import_ref Main//b, loc7_32, loaded [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness.243)]
// CHECK:STDOUT:   %Main.import_ref.509: @C.as.I.impl.%C.as.I.impl.F.type.2 (%C.as.I.impl.F.type.029cdf.1) = import_ref Main//b, loc8_24, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F.2 (constants.%C.as.I.impl.F.49b6ba.1)]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.509), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.2: %empty_tuple.type = import_ref Main//b, loc7_15, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.F.811: @C.as.I.impl.%C.as.I.impl.F.type.1 (%C.as.I.impl.F.type.029cdf.2) = import_ref Main//b, F, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F.1 (constants.%C.as.I.impl.F.49b6ba.2)]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.3: %empty_tuple.type = import_ref Main//b, loc7_15, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.import_ref.d34: type = import_ref Main//b, loc7_25, loaded [symbolic = @C.as.I.impl.%C (constants.%C.9015c3.2)]
// CHECK:STDOUT:   %Main.import_ref.ca3: type = import_ref Main//b, loc7_30, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.4: %empty_tuple.type = import_ref Main//b, loc7_15, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core.a3bffd.1
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.366
// CHECK:STDOUT:   .F = imports.%Main.import_ref.7f7
// CHECK:STDOUT:   witness = (imports.%Main.F.09a)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(imports.%Main.import_ref.c90e1a.4: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.9015c3.2)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic = %I.impl_witness (constants.%I.impl_witness.243)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type.1 (constants.%C.as.I.impl.F.type.029cdf.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.1: @C.as.I.impl.%C.as.I.impl.F.type.1 (%C.as.I.impl.F.type.029cdf.2) = struct_value () [symbolic = %C.as.I.impl.F.1 (constants.%C.as.I.impl.F.49b6ba.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.type.2: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type.2 (constants.%C.as.I.impl.F.type.029cdf.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.2: @C.as.I.impl.%C.as.I.impl.F.type.2 (%C.as.I.impl.F.type.029cdf.1) = struct_value () [symbolic = %C.as.I.impl.F.2 (constants.%C.as.I.impl.F.49b6ba.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.d34 as imports.%Main.import_ref.ca3 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.867
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
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.602
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc7_6: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_7: %empty_tuple.type = converted %.loc7_6, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_tuple) [concrete = constants.%C.144]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref.loc7_11: %I.assoc_type = name_ref F, imports.%Main.import_ref.7f7 [concrete = constants.%assoc0.c00]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (constants.%I.impl_witness.b9d) [concrete = constants.%I.facet.bae]
// CHECK:STDOUT:   %.loc7_8: %I.type = converted %C, %I.facet [concrete = constants.%I.facet.bae]
// CHECK:STDOUT:   %impl.elem0: %.612 = impl_witness_access constants.%I.impl_witness.b9d, element0 [concrete = constants.%C.as.I.impl.F.2e70f7.2]
// CHECK:STDOUT:   %.loc7_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %empty_struct.loc7_16.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17.1: %empty_struct_type = converted %.loc7_16.1, %empty_struct.loc7_16.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.I.impl.F.1(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.specific_fn.157186.1]
// CHECK:STDOUT:   %empty_struct.loc7_16.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_16.2: %empty_struct_type = converted %.loc7_16.1, %empty_struct.loc7_16.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17.2: %C.as.I.impl.F.type.0f4092.1 = specific_constant imports.%Main.F.811, @C.as.I.impl(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.2e70f7.1]
// CHECK:STDOUT:   %F.ref.loc7_17: %C.as.I.impl.F.type.0f4092.1 = name_ref F, %.loc7_17.2 [concrete = constants.%C.as.I.impl.F.2e70f7.1]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %F.ref.loc7_17, @C.as.I.impl.F.2(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.specific_fn.157186.2]
// CHECK:STDOUT:   %.loc7_16.3: ref %C.144 = temporary_storage
// CHECK:STDOUT:   %.loc7_16.4: init %C.144 to %.loc7_16.3 = class_init () [concrete = constants.%C.val.2c6]
// CHECK:STDOUT:   %.loc7_16.5: init %C.144 = converted %.loc7_16.2, %.loc7_16.4 [concrete = constants.%C.val.2c6]
// CHECK:STDOUT:   %.loc7_16.6: ref %C.144 = temporary %.loc7_16.3, %.loc7_16.5 [concrete = constants.%.6f8]
// CHECK:STDOUT:   %.loc7_16.7: %C.144 = acquire_value %.loc7_16.6 [concrete = constants.%C.val.2c6]
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %C.as.I.impl.F.specific_fn(%.loc7_16.7)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.6f8)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.f4a753.2: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.1(imports.%Main.import_ref.c90e1a.3: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.029cdf.2)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.F.1.%C.as.I.impl.F.type (%C.as.I.impl.F.type.029cdf.2) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.49b6ba.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F, @C.as.I.impl.F.2(%Y) [symbolic = %C.as.I.impl.F.specific_fn (constants.%C.as.I.impl.F.specific_fn.219)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.9015c3.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %C.val: @C.as.I.impl.F.1.%C (%C.9015c3.2) = struct_value () [symbolic = %C.val (constants.%C.val.130)]
// CHECK:STDOUT:   %.1: ref @C.as.I.impl.F.1.%C (%C.9015c3.2) = temporary invalid, %C.val [symbolic = %.1 (constants.%.af3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.33b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b7e)]
// CHECK:STDOUT:   %.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.2 (constants.%.7e5)]
// CHECK:STDOUT:   %impl.elem0: @C.as.I.impl.F.1.%.2 (%.7e5) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0)]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.1, %impl.elem0 [symbolic = %bound_method.1 (constants.%bound_method.c0c)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.1, %specific_impl_fn [symbolic = %bound_method.2 (constants.%bound_method.e8f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn [thunk imports.%Main.F.811 for imports.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.a37)];
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.2(imports.%Main.import_ref.c90e1a.2: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.9015c3.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.556)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_16.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_16.2(%self.param: ref %C.144) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.c30) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c30
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.de0
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.8e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.c30) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y) {
// CHECK:STDOUT:   %X => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.9015c3.2
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.243
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1 => constants.%C.as.I.impl.F.type.029cdf.2
// CHECK:STDOUT:   %C.as.I.impl.F.1 => constants.%C.as.I.impl.F.49b6ba.2
// CHECK:STDOUT:   %C.as.I.impl.F.type.2 => constants.%C.as.I.impl.F.type.029cdf.1
// CHECK:STDOUT:   %C.as.I.impl.F.2 => constants.%C.as.I.impl.F.49b6ba.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.2(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.9015c3.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.556
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.a37) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.1(constants.%Y) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.144
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.b9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1 => constants.%C.as.I.impl.F.type.0f4092.1
// CHECK:STDOUT:   %C.as.I.impl.F.1 => constants.%C.as.I.impl.F.2e70f7.1
// CHECK:STDOUT:   %C.as.I.impl.F.type.2 => constants.%C.as.I.impl.F.type.0f4092.2
// CHECK:STDOUT:   %C.as.I.impl.F.2 => constants.%C.as.I.impl.F.2e70f7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.bae) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.bae
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.be4
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ea9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.1(constants.%empty_tuple) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C.as.I.impl.F.type => constants.%C.as.I.impl.F.type.0f4092.1
// CHECK:STDOUT:   %C.as.I.impl.F => constants.%C.as.I.impl.F.2e70f7.1
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn => constants.%C.as.I.impl.F.specific_fn.157186.2
// CHECK:STDOUT:   %C => constants.%C.144
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %C.val => constants.%C.val.2c6
// CHECK:STDOUT:   %.1 => constants.%.6f8
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1367bd.2
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.06b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.417
// CHECK:STDOUT:   %.2 => constants.%.6e3
// CHECK:STDOUT:   %impl.elem0 => constants.%Destroy.Op.f0ba5f.2
// CHECK:STDOUT:   %bound_method.1 => constants.%Destroy.Op.bound
// CHECK:STDOUT:   %specific_impl_fn => constants.%Destroy.Op.f0ba5f.2
// CHECK:STDOUT:   %bound_method.2 => constants.%Destroy.Op.bound
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.2(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.144
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.02a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
