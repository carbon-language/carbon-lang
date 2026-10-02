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

class C(X: ()) {}

impl forall [Y: ()] C(Y) as I {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc5_12.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = wrapper_binding x, %x.param
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %X.patt: %pattern_type.cb1 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %X: %empty_tuple.type = symbolic_binding X, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.092206.1: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [symbolic]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic]
// CHECK:STDOUT:   %C.092206.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.dda: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.eae: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.dda) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.4b1: %I.WithSelf.F.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %x.param_patt.c77: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.36b: %pattern_type.a96 = wrapper_binding_pattern x, %x.param_patt.c77 [concrete]
// CHECK:STDOUT:   %.b6e: <witness> = impl_self_witness %C.092206.2, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %pattern_type.dc9: type = pattern_type %C.092206.2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.127: %pattern_type.dc9 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.9e9: %pattern_type.dc9 = wrapper_binding_pattern x, %x.param_patt.127 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.f7d071.1: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.4d196e.1: %C.as.I.impl.F.type.f7d071.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.092206.2, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.1dd: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.f71: %I.WithSelf.F.type.1dd = struct_value () [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.f7d071.2: type = fn_type @C.as.I.impl.F.loc8_24.2, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.4d196e.2: %C.as.I.impl.F.type.f7d071.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.f3a: <witness> = require_complete_type %C.092206.2 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F.4d196e.1, @C.as.I.impl.F.loc8_24.1(%Y) [symbolic]
// CHECK:STDOUT:   %C.val: %C.092206.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2b3: ref %C.092206.2 = temporary invalid, %C.val [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.869: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.c97: %Destroy.WithSelf.SelfDestruct.type.869 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.df0 [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.cf1: <witness> = lookup_impl_witness %C.092206.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.763: %Destroy.type = facet_value %C.092206.2, (%Destroy.lookup_impl_witness.cf1) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.0b7: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.763) [symbolic]
// CHECK:STDOUT:   %.488: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.0b7, %Destroy.facet.763 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.488 = impl_witness_access %Destroy.lookup_impl_witness.cf1, element2 [symbolic]
// CHECK:STDOUT:   %bound_method.62e: <bound method> = bound_method %.2b3, %impl.elem2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f81: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.763) [symbolic]
// CHECK:STDOUT:   %bound_method.d8c: <bound method> = bound_method %.2b3, %specific_impl_fn.f81 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.e44 = import_ref Main//a, loc5_14, unloaded
// CHECK:STDOUT:   %Main.F: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.eae) = import_ref Main//a, F, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.4b1)]
// CHECK:STDOUT:   %I.WithSelf.F.decl: %I.WithSelf.F.type.eae = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.4b1)] {} {}
// CHECK:STDOUT:   %Main.import_ref.e59c42.2: %I.type = import_ref Main//a, loc4_13, loaded [symbolic = constants.%Self.dda]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.8ba: %Destroy.assoc_type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.df0: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.869) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.c97)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.2 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %X.patt.loc5_10.1: %pattern_type.cb1 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc5_10.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_13.1: type = splice_block %.loc5_13.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_13.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc5_13.3: type = converted %.loc5_13.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc5_10.2: %empty_tuple.type = symbolic_binding X, 0 [symbolic = %X.loc5_10.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %Y.patt.loc7_15.1: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [symbolic = %Y.patt.loc7_15.2 (constants.%Y.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %Y.ref: %empty_tuple.type = name_ref Y, %Y.loc7_15.2 [symbolic = %Y.loc7_15.1 (constants.%Y)]
// CHECK:STDOUT:     %C.loc7_24.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc7_24.1 (constants.%C.092206.2)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_18.1: type = splice_block %.loc7_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc7_18.3: type = converted %.loc7_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.loc7_15.2: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y.loc7_15.1 (constants.%Y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.I.impl.%C.loc7_24.2, @I [symbolic = @C.as.I.impl.%.loc7_31 (constants.%.b6e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .F = imports.%Main.import_ref.e44
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%Y.loc7_15.2: %empty_tuple.type) {
// CHECK:STDOUT:   %Y.patt.loc7_15.2: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [symbolic = %Y.patt.loc7_15.2 (constants.%Y.patt)]
// CHECK:STDOUT:   %Y.loc7_15.1: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y.loc7_15.1 (constants.%Y)]
// CHECK:STDOUT:   %C.loc7_24.1: type = class_type @C, @C(%Y.loc7_15.1) [symbolic = %C.loc7_24.1 (constants.%C.092206.2)]
// CHECK:STDOUT:   %.loc7_31: <witness> = impl_self_witness %C.loc7_24.1, @I [symbolic = %.loc7_31 (constants.%.b6e)]
// CHECK:STDOUT:   %I.impl_witness.loc7_31.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %I.impl_witness.loc7_31.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.1: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %C.as.I.impl.F.type.loc8_24.1 (constants.%C.as.I.impl.F.type.f7d071.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.1: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.1 (%C.as.I.impl.F.type.f7d071.1) = struct_value () [symbolic = %C.as.I.impl.F.loc8_24.1 (constants.%C.as.I.impl.F.4d196e.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.2: type = fn_type @C.as.I.impl.F.loc8_24.2, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %C.as.I.impl.F.type.loc8_24.2 (constants.%C.as.I.impl.F.type.f7d071.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.2: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.2 (%C.as.I.impl.F.type.f7d071.2) = struct_value () [symbolic = %C.as.I.impl.F.loc8_24.2 (constants.%C.as.I.impl.F.4d196e.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_24.2 as %I.ref {
// CHECK:STDOUT:     %C.as.I.impl.F.decl.loc8_24.1: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.1 (%C.as.I.impl.F.type.f7d071.1) = fn_decl @C.as.I.impl.F.loc8_24.1 [symbolic = @C.as.I.impl.%C.as.I.impl.F.loc8_24.1 (constants.%C.as.I.impl.F.4d196e.1)] {
// CHECK:STDOUT:       %x.param_patt.loc8_16.1: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.dc9) = value_param_pattern [symbolic = %x.param_patt.loc8_16.2 (constants.%x.param_patt.127)]
// CHECK:STDOUT:       %x.patt.loc8_16.1: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.dc9) = wrapper_binding_pattern x, %x.param_patt.loc8_16.1 [symbolic = %x.patt.loc8_16.2 (constants.%x.patt.9e9)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.092206.2) = value_param call_param0
// CHECK:STDOUT:       %.loc8: type = splice_block %C.loc8_21.2 [symbolic = %C.loc8_21.1 (constants.%C.092206.2)] {
// CHECK:STDOUT:         %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:         %Y.ref: %empty_tuple.type = name_ref Y, @C.as.I.impl.%Y.loc7_15.2 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:         %C.loc8_21.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc8_21.1 (constants.%C.092206.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.092206.2) = wrapper_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.as.I.impl.F.decl.loc8_24.2: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.2 (%C.as.I.impl.F.type.f7d071.2) = fn_decl @C.as.I.impl.F.loc8_24.2 [symbolic = @C.as.I.impl.%C.as.I.impl.F.loc8_24.2 (constants.%C.as.I.impl.F.4d196e.2)] {
// CHECK:STDOUT:       %.1: %pattern_type.a96 = specific_constant constants.%x.param_patt.c77, @I.WithSelf.F(constants.%I.facet) [concrete = constants.%x.param_patt.c77]
// CHECK:STDOUT:       %x.patt: %pattern_type.a96 = wrapper_binding_pattern x, %.1 [concrete = constants.%x.patt.36b]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:       %x: %empty_struct_type = wrapper_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl.loc8_24.2), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_31.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%Y) [symbolic = %I.impl_witness.loc7_31.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .Y = <poisoned>
// CHECK:STDOUT:     .F = %C.as.I.impl.F.decl.loc8_24.1
// CHECK:STDOUT:     extend %I.ref
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_31.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%X.loc5_10.2: %empty_tuple.type) {
// CHECK:STDOUT:   %X.patt.loc5_10.2: %pattern_type.cb1 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc5_10.2 (constants.%X.patt)]
// CHECK:STDOUT:   %X.loc5_10.1: %empty_tuple.type = symbolic_binding X, 0 [symbolic = %X.loc5_10.1 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.092206.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.e59c42.2: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.loc8_24.1(@C.as.I.impl.%Y.loc7_15.2: %empty_tuple.type) {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.loc8_21.1: type = class_type @C, @C(%Y) [symbolic = %C.loc8_21.1 (constants.%C.092206.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc8_21.1 [symbolic = %pattern_type (constants.%pattern_type.dc9)]
// CHECK:STDOUT:   %x.param_patt.loc8_16.2: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.dc9) = value_param_pattern [symbolic = %x.param_patt.loc8_16.2 (constants.%x.param_patt.127)]
// CHECK:STDOUT:   %x.patt.loc8_16.2: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.dc9) = wrapper_binding_pattern x, %x.param_patt.loc8_16.2 [symbolic = %x.patt.loc8_16.2 (constants.%x.patt.9e9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc8_21.1 [symbolic = %require_complete (constants.%require_complete.f3a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.092206.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.loc8_24.2(@C.as.I.impl.%Y.loc7_15.2: %empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.f7d071.1)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.f7d071.1) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.4d196e.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.loc8_24.2: <specific function> = specific_function %C.as.I.impl.F, @C.as.I.impl.F.loc8_24.1(%Y) [symbolic = %C.as.I.impl.F.specific_fn.loc8_24.2 (constants.%C.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.092206.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete.f3a)]
// CHECK:STDOUT:   %C.val: @C.as.I.impl.F.loc8_24.2.%C (%C.092206.2) = struct_value () [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:   %.9: ref @C.as.I.impl.F.loc8_24.2.%C (%C.092206.2) = temporary invalid, %C.val [symbolic = %.9 (constants.%.2b3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.cf1)]
// CHECK:STDOUT:   %Destroy.facet.3: %Destroy.type = facet_value %C, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.763)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.0b7)]
// CHECK:STDOUT:   %.10: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.3 [symbolic = %.10 (constants.%.488)]
// CHECK:STDOUT:   %impl.elem2.2: @C.as.I.impl.F.loc8_24.2.%.10 (%.488) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %.9, %impl.elem2.2 [symbolic = %bound_method.3 (constants.%bound_method.62e)]
// CHECK:STDOUT:   %specific_impl_fn.2: <specific function> = specific_impl_function %impl.elem2.2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.3) [symbolic = %specific_impl_fn.2 (constants.%specific_impl_fn.f81)]
// CHECK:STDOUT:   %bound_method.4: <bound method> = bound_method %.9, %specific_impl_fn.2 [symbolic = %bound_method.4 (constants.%bound_method.d8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: %empty_struct_type) [thunk @C.as.I.impl.%C.as.I.impl.F.decl.loc8_24.1 for imports.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet)] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.f7d071.1) = specific_constant @C.as.I.impl.%C.as.I.impl.F.decl.loc8_24.1, @C.as.I.impl(constants.%Y) [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.4d196e.1)]
// CHECK:STDOUT:     %F.ref: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.f7d071.1) = name_ref F, %.loc8 [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.4d196e.1)]
// CHECK:STDOUT:     %C.as.I.impl.F.specific_fn.loc8_24.1: <specific function> = specific_function %F.ref, @C.as.I.impl.F.loc8_24.1(constants.%Y) [symbolic = %C.as.I.impl.F.specific_fn.loc8_24.2 (constants.%C.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %.2: ref @C.as.I.impl.F.loc8_24.2.%C (%C.092206.2) = temporary_storage
// CHECK:STDOUT:     %.3: init @C.as.I.impl.F.loc8_24.2.%C (%C.092206.2) to %.2 = class_init () [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %.4: init @C.as.I.impl.F.loc8_24.2.%C (%C.092206.2) = converted %x.param, %.3 [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %.5: ref @C.as.I.impl.F.loc8_24.2.%C (%C.092206.2) = temporary %.2, %.4 [symbolic = %.9 (constants.%.2b3)]
// CHECK:STDOUT:     %.6: @C.as.I.impl.F.loc8_24.2.%C (%C.092206.2) = acquire_value %.5 [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %C.as.I.impl.F.call: init %empty_tuple.type = call %C.as.I.impl.F.specific_fn.loc8_24.1(%.6)
// CHECK:STDOUT:     %SelfDestruct.ref: %Destroy.assoc_type = name_ref SelfDestruct, imports.%Core.import_ref.8ba [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.1: @C.as.I.impl.F.loc8_24.2.%.10 (%.488) = impl_witness_access constants.%Destroy.lookup_impl_witness.cf1, element2 [symbolic = %impl.elem2.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.1: <bound method> = bound_method %.5, %impl.elem2.1 [symbolic = %bound_method.3 (constants.%bound_method.62e)]
// CHECK:STDOUT:     %Destroy.facet.1: %Destroy.type = facet_value constants.%C.092206.2, (constants.%Destroy.lookup_impl_witness.cf1) [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.763)]
// CHECK:STDOUT:     %.7: %Destroy.type = converted constants.%C.092206.2, %Destroy.facet.1 [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.763)]
// CHECK:STDOUT:     %Destroy.facet.2: %Destroy.type = facet_value constants.%C.092206.2, (constants.%Destroy.lookup_impl_witness.cf1) [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.763)]
// CHECK:STDOUT:     %.8: %Destroy.type = converted constants.%C.092206.2, %Destroy.facet.2 [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.763)]
// CHECK:STDOUT:     %specific_impl_fn.1: <specific function> = specific_impl_function %impl.elem2.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Destroy.facet.763) [symbolic = %specific_impl_fn.2 (constants.%specific_impl_fn.f81)]
// CHECK:STDOUT:     %bound_method.2: <bound method> = bound_method %.5, %specific_impl_fn.1 [symbolic = %bound_method.4 (constants.%bound_method.d8c)]
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.2(%.5)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.2(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.patt.loc5_10.2 => constants.%X.patt
// CHECK:STDOUT:   %X.loc5_10.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y) {
// CHECK:STDOUT:   %X.patt.loc5_10.2 => constants.%X.patt
// CHECK:STDOUT:   %X.loc5_10.1 => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.dda) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.dda
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.eae
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.4b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.dda) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Y) {
// CHECK:STDOUT:   %Y.patt.loc7_15.2 => constants.%Y.patt
// CHECK:STDOUT:   %Y.loc7_15.1 => constants.%Y
// CHECK:STDOUT:   %C.loc7_24.1 => constants.%C.092206.2
// CHECK:STDOUT:   %.loc7_31 => constants.%.b6e
// CHECK:STDOUT:   %I.impl_witness.loc7_31.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.1 => constants.%C.as.I.impl.F.type.f7d071.1
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.1 => constants.%C.as.I.impl.F.4d196e.1
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.2 => constants.%C.as.I.impl.F.type.f7d071.2
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.2 => constants.%C.as.I.impl.F.4d196e.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.loc8_24.1(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C.loc8_21.1 => constants.%C.092206.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dc9
// CHECK:STDOUT:   %x.param_patt.loc8_16.2 => constants.%x.param_patt.127
// CHECK:STDOUT:   %x.patt.loc8_16.2 => constants.%x.patt.9e9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.f3a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.1dd
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.f71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.loc8_24.2(constants.%Y) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: %empty_tuple.type = symbolic_binding X, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %X.patt: %pattern_type.cb1 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.e03: type = class_type @C, @C(%empty_tuple) [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.dda: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.eae: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.dda) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.4b1: %I.WithSelf.F.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c77 [concrete]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic]
// CHECK:STDOUT:   %C.092206.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %.b6e: <witness> = impl_self_witness %C.092206.2, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.719: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.9d343d.1: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.d3a8b5.1: %C.as.I.impl.F.type.9d343d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.9d343d.2: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.d3a8b5.2: %C.as.I.impl.F.type.9d343d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.dc9: type = pattern_type %C.092206.2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.127: %pattern_type.dc9 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [symbolic]
// CHECK:STDOUT:   %x.patt.9e9: %pattern_type.dc9 = wrapper_binding_pattern x, %x.param_patt.127 [symbolic]
// CHECK:STDOUT:   %require_complete.f3a: <witness> = require_complete_type %C.092206.2 [symbolic]
// CHECK:STDOUT:   %I.facet.83a: %I.type = facet_value %C.092206.2, (%I.impl_witness.719) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.cf1: <witness> = lookup_impl_witness %C.092206.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.763: %Destroy.type = facet_value %C.092206.2, (%Destroy.lookup_impl_witness.cf1) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.0b7: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.763) [symbolic]
// CHECK:STDOUT:   %.488: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.0b7, %Destroy.facet.763 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.488 = impl_witness_access %Destroy.lookup_impl_witness.cf1, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f81: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.763) [symbolic]
// CHECK:STDOUT:   %C.val.c72: %C.092206.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2b3: ref %C.092206.2 = temporary invalid, %C.val.c72 [symbolic]
// CHECK:STDOUT:   %bound_method.d8c: <bound method> = bound_method %.2b3, %specific_impl_fn.f81 [symbolic]
// CHECK:STDOUT:   %bound_method.62e: <bound method> = bound_method %.2b3, %impl.elem2 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.dc7: <specific function> = specific_function %C.as.I.impl.F.d3a8b5.2, @C.as.I.impl.F.2(%Y) [symbolic]
// CHECK:STDOUT:   %.bd8: <witness> = impl_self_witness %C.e03, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.66a: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.e38bfd.1: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.fcdf5c.1: %C.as.I.impl.F.type.e38bfd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.e38bfd.2: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.fcdf5c.2: %C.as.I.impl.F.type.e38bfd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.8cc: %I.type = facet_value %C.e03, (%I.impl_witness.66a) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.0a5: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.8cc) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ae9: %I.WithSelf.F.type.0a5 = struct_value () [concrete]
// CHECK:STDOUT:   %.a01: type = fn_type_with_self_type %I.WithSelf.F.type.0a5, %I.facet.8cc [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.6f6417.1: <specific function> = specific_function %C.as.I.impl.F.fcdf5c.2, @C.as.I.impl.F.1(%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.90c: type = pattern_type %C.e03 [concrete]
// CHECK:STDOUT:   %x.param_patt.b88: %pattern_type.90c = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.84b: %pattern_type.90c = wrapper_binding_pattern x, %x.param_patt.b88 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.6f6417.2: <specific function> = specific_function %C.as.I.impl.F.fcdf5c.1, @C.as.I.impl.F.2(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.val.f1d: %C.e03 = struct_value () [concrete]
// CHECK:STDOUT:   %.5b8: ref %C.e03 = temporary invalid, %C.val.f1d [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc7_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.661: %pattern_type.90c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.fd4: %pattern_type.90c = wrapper_binding_pattern self, %self.param_patt.661 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc7_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc7_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %Destroy.WithSelf.SubobjectDestroy.d01daf.2, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.68e: %Destroy.type = facet_value %C.e03, (%custom_witness.f8f19d.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e98: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.68e) [concrete]
// CHECK:STDOUT:   %.c7f: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.e98, %Destroy.facet.68e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.5b8, %Destroy.WithSelf.SelfDestruct.db3fdb.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//b, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//b, loc5_17, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.418 = import_ref Main//b, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c90e1a.1: %empty_tuple.type = import_ref Main//b, loc5_10, loaded [symbolic = @C.%X (constants.%X)]
// CHECK:STDOUT:   %Main.import_ref.a45: %I.assoc_type = import_ref Main//a, loc5_14, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.F.6b8 = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %I.WithSelf.F.decl: %I.WithSelf.F.type.eae = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.4b1)] {} {}
// CHECK:STDOUT:   %Main.import_ref.e59c42.2: %I.type = import_ref Main//a, loc4_13, loaded [symbolic = constants.%Self.dda]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c77: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.eae) = import_ref Main//a, loc5_14, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.4b1)]
// CHECK:STDOUT:   %Main.import_ref.30c: <witness> = import_ref Main//b, loc7_31, loaded [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness.719)]
// CHECK:STDOUT:   %Main.import_ref.72d: @C.as.I.impl.%C.as.I.impl.F.type.2 (%C.as.I.impl.F.type.9d343d.1) = import_ref Main//b, loc8_24, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F.2 (constants.%C.as.I.impl.F.d3a8b5.1)]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.72d), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.2: %empty_tuple.type = import_ref Main//b, loc7_15, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.F.fa7: @C.as.I.impl.%C.as.I.impl.F.type.1 (%C.as.I.impl.F.type.9d343d.2) = import_ref Main//b, F, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F.1 (constants.%C.as.I.impl.F.d3a8b5.2)]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.3: %empty_tuple.type = import_ref Main//b, loc7_15, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.import_ref.b9a: type = import_ref Main//b, loc7_24, loaded [symbolic = @C.as.I.impl.%C (constants.%C.092206.2)]
// CHECK:STDOUT:   %Main.import_ref.e90: type = import_ref Main//b, loc7_29, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.4: %empty_tuple.type = import_ref Main//b, loc7_15, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_16.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc7_16.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_16.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.90c = ref_param_pattern [concrete = constants.%self.param_patt.661]
// CHECK:STDOUT:     %self.patt: %pattern_type.90c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fd4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.e03 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.e03 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc7_16.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.90c = ref_param_pattern [concrete = constants.%self.param_patt.661]
// CHECK:STDOUT:     %self.patt: %pattern_type.90c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fd4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.e03 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.e03 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .F = imports.%Main.import_ref.a45
// CHECK:STDOUT:   witness = (imports.%Main.F.6b8)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(imports.%Main.import_ref.c90e1a.4: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [symbolic = %Y.patt (constants.%Y.patt)]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.092206.2)]
// CHECK:STDOUT:   %.1: <witness> = impl_self_witness %C, @I [symbolic = %.1 (constants.%.b6e)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic = %I.impl_witness (constants.%I.impl_witness.719)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type.1 (constants.%C.as.I.impl.F.type.9d343d.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.1: @C.as.I.impl.%C.as.I.impl.F.type.1 (%C.as.I.impl.F.type.9d343d.2) = struct_value () [symbolic = %C.as.I.impl.F.1 (constants.%C.as.I.impl.F.d3a8b5.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.type.2: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type.2 (constants.%C.as.I.impl.F.type.9d343d.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.2: @C.as.I.impl.%C.as.I.impl.F.type.2 (%C.as.I.impl.F.type.9d343d.1) = struct_value () [symbolic = %C.as.I.impl.F.2 (constants.%C.as.I.impl.F.d3a8b5.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.b9a as imports.%Main.import_ref.e90 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.30c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.c90e1a.1: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %X.patt: %pattern_type.cb1 = symbolic_binding_pattern X, 0 [symbolic = %X.patt (constants.%X.patt)]
// CHECK:STDOUT:   %X: %empty_tuple.type = symbolic_binding X, 0 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.418
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc7_6: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_7: %empty_tuple.type = converted %.loc7_6, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_tuple) [concrete = constants.%C.e03]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref.loc7_11: %I.assoc_type = name_ref F, imports.%Main.import_ref.a45 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (constants.%I.impl_witness.66a) [concrete = constants.%I.facet.8cc]
// CHECK:STDOUT:   %.loc7_8: %I.type = converted %C, %I.facet [concrete = constants.%I.facet.8cc]
// CHECK:STDOUT:   %impl.elem0: %.a01 = impl_witness_access constants.%I.impl_witness.66a, element0 [concrete = constants.%C.as.I.impl.F.fcdf5c.2]
// CHECK:STDOUT:   %.loc7_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.I.impl.F.1(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.specific_fn.6f6417.1]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_16.2: %empty_struct_type = converted %.loc7_16.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17: %C.as.I.impl.F.type.e38bfd.1 = specific_constant imports.%Main.F.fa7, @C.as.I.impl(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.fcdf5c.1]
// CHECK:STDOUT:   %F.ref.loc7_17: %C.as.I.impl.F.type.e38bfd.1 = name_ref F, %.loc7_17 [concrete = constants.%C.as.I.impl.F.fcdf5c.1]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %F.ref.loc7_17, @C.as.I.impl.F.2(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.specific_fn.6f6417.2]
// CHECK:STDOUT:   %.loc7_16.3: ref %C.e03 = temporary_storage
// CHECK:STDOUT:   %.loc7_16.4: init %C.e03 to %.loc7_16.3 = class_init () [concrete = constants.%C.val.f1d]
// CHECK:STDOUT:   %.loc7_16.5: init %C.e03 = converted %.loc7_16.2, %.loc7_16.4 [concrete = constants.%C.val.f1d]
// CHECK:STDOUT:   %.loc7_16.6: ref %C.e03 = temporary %.loc7_16.3, %.loc7_16.5 [concrete = constants.%.5b8]
// CHECK:STDOUT:   %.loc7_16.7: %C.e03 = acquire_value %.loc7_16.6 [concrete = constants.%C.val.f1d]
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %C.as.I.impl.F.specific_fn(%.loc7_16.7)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.5b8)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.e59c42.2: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.1(imports.%Main.import_ref.c90e1a.3: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.9d343d.2)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.F.1.%C.as.I.impl.F.type (%C.as.I.impl.F.type.9d343d.2) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.d3a8b5.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F, @C.as.I.impl.F.2(%Y) [symbolic = %C.as.I.impl.F.specific_fn (constants.%C.as.I.impl.F.specific_fn.dc7)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.092206.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete.f3a)]
// CHECK:STDOUT:   %C.val: @C.as.I.impl.F.1.%C (%C.092206.2) = struct_value () [symbolic = %C.val (constants.%C.val.c72)]
// CHECK:STDOUT:   %.1: ref @C.as.I.impl.F.1.%C (%C.092206.2) = temporary invalid, %C.val [symbolic = %.1 (constants.%.2b3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.cf1)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.763)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.0b7)]
// CHECK:STDOUT:   %.2: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet [symbolic = %.2 (constants.%.488)]
// CHECK:STDOUT:   %impl.elem2: @C.as.I.impl.F.1.%.2 (%.488) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.1, %impl.elem2 [symbolic = %bound_method.1 (constants.%bound_method.62e)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.f81)]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.1, %specific_impl_fn [symbolic = %bound_method.2 (constants.%bound_method.d8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn [thunk imports.%Main.F.fa7 for imports.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.83a)];
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.2(imports.%Main.import_ref.c90e1a.2: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.092206.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.dc9)]
// CHECK:STDOUT:   %x.param_patt: @C.as.I.impl.F.2.%pattern_type (%pattern_type.dc9) = value_param_pattern [symbolic = %x.param_patt (constants.%x.param_patt.127)]
// CHECK:STDOUT:   %x.patt: @C.as.I.impl.F.2.%pattern_type (%pattern_type.dc9) = wrapper_binding_pattern x, %x.param_patt [symbolic = %x.patt (constants.%x.patt.9e9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete.f3a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_16.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_16.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_16.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_16.2(%self.param: ref %C.e03) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_16.2(%self.param: ref %C.e03) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_16.2(%self.param: ref %C.e03) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %X.patt => constants.%X.patt
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_tuple) {
// CHECK:STDOUT:   %X.patt => constants.%X.patt
// CHECK:STDOUT:   %X => constants.%empty_tuple
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.dda) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.dda
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.eae
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.4b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.dda) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Y) {
// CHECK:STDOUT:   %X.patt => constants.%X.patt
// CHECK:STDOUT:   %X => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Y) {
// CHECK:STDOUT:   %Y.patt => constants.%Y.patt
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.092206.2
// CHECK:STDOUT:   %.1 => constants.%.b6e
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.719
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1 => constants.%C.as.I.impl.F.type.9d343d.2
// CHECK:STDOUT:   %C.as.I.impl.F.1 => constants.%C.as.I.impl.F.d3a8b5.2
// CHECK:STDOUT:   %C.as.I.impl.F.type.2 => constants.%C.as.I.impl.F.type.9d343d.1
// CHECK:STDOUT:   %C.as.I.impl.F.2 => constants.%C.as.I.impl.F.d3a8b5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.2(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.092206.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dc9
// CHECK:STDOUT:   %x.param_patt => constants.%x.param_patt.127
// CHECK:STDOUT:   %x.patt => constants.%x.patt.9e9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.f3a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.83a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.1(constants.%Y) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y.patt => constants.%Y.patt
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.e03
// CHECK:STDOUT:   %.1 => constants.%.bd8
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.66a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1 => constants.%C.as.I.impl.F.type.e38bfd.1
// CHECK:STDOUT:   %C.as.I.impl.F.1 => constants.%C.as.I.impl.F.fcdf5c.1
// CHECK:STDOUT:   %C.as.I.impl.F.type.2 => constants.%C.as.I.impl.F.type.e38bfd.2
// CHECK:STDOUT:   %C.as.I.impl.F.2 => constants.%C.as.I.impl.F.fcdf5c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.8cc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.8cc
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.0a5
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ae9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.1(constants.%empty_tuple) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C.as.I.impl.F.type => constants.%C.as.I.impl.F.type.e38bfd.1
// CHECK:STDOUT:   %C.as.I.impl.F => constants.%C.as.I.impl.F.fcdf5c.1
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn => constants.%C.as.I.impl.F.specific_fn.6f6417.2
// CHECK:STDOUT:   %C => constants.%C.e03
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %C.val => constants.%C.val.f1d
// CHECK:STDOUT:   %.1 => constants.%.5b8
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.2
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.68e
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.e98
// CHECK:STDOUT:   %.2 => constants.%.c7f
// CHECK:STDOUT:   %impl.elem2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %bound_method.1 => constants.%Destroy.WithSelf.SelfDestruct.bound
// CHECK:STDOUT:   %specific_impl_fn => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %bound_method.2 => constants.%Destroy.WithSelf.SelfDestruct.bound
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.2(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.e03
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.90c
// CHECK:STDOUT:   %x.param_patt => constants.%x.param_patt.b88
// CHECK:STDOUT:   %x.patt => constants.%x.patt.84b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
