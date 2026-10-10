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
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.685960.1: type = class_type @C, @C(%X) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [symbolic]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic]
// CHECK:STDOUT:   %C.685960.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.dda: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.418: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.dda) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.ca5: %I.WithSelf.F.type.418 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %x.param_patt.c77: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.36b: %pattern_type.a96 = wrapper_binding_pattern x, %x.param_patt.c77 [concrete]
// CHECK:STDOUT:   %.243: <witness> = impl_self_witness %C.685960.2, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %pattern_type.a22: type = pattern_type %C.685960.2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.7af: %pattern_type.a22 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.574: %pattern_type.a22 = wrapper_binding_pattern x, %x.param_patt.7af [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.95b168.1: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.bf08b1.1: %C.as.I.impl.F.type.95b168.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C.685960.2, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.7c0: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.11d: %I.WithSelf.F.type.7c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.95b168.2: type = fn_type @C.as.I.impl.F.loc8_24.2, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.bf08b1.2: %C.as.I.impl.F.type.95b168.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.544: <witness> = require_complete_type %C.685960.2 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F.bf08b1.1, @C.as.I.impl.F.loc8_24.1(%Y) [symbolic]
// CHECK:STDOUT:   %C.val: %C.685960.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.cba: ref %C.685960.2 = temporary invalid, %C.val [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4e5: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.4a0: %Destroy.WithSelf.SelfDestruct.type.4e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.71d [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.490: <witness> = lookup_impl_witness %C.685960.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.bf5: %Destroy.type = facet_value %C.685960.2, (%Destroy.lookup_impl_witness.490) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.a37: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.bf5) [symbolic]
// CHECK:STDOUT:   %.513: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.a37, %Destroy.facet.bf5 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.513 = impl_witness_access %Destroy.lookup_impl_witness.490, element2 [symbolic]
// CHECK:STDOUT:   %bound_method.f09: <bound method> = bound_method %.cba, %impl.elem2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.855: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.bf5) [symbolic]
// CHECK:STDOUT:   %bound_method.6af: <bound method> = bound_method %.cba, %specific_impl_fn.855 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//a, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.046 = import_ref Main//a, loc5_14, unloaded
// CHECK:STDOUT:   %Main.F: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.418) = import_ref Main//a, F, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ca5)]
// CHECK:STDOUT:   %I.WithSelf.F.decl: %I.WithSelf.F.type.418 = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ca5)] {} {}
// CHECK:STDOUT:   %Main.import_ref.e59c42.2: %I.type = import_ref Main//a, loc4_13, loaded [symbolic = constants.%Self.dda]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.e39: %Destroy.assoc_type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.71d: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.4e5) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.4a0)]
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
// CHECK:STDOUT:     %C.loc7_24.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc7_24.1 (constants.%C.685960.2)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc7_18.1: type = splice_block %.loc7_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc7_18.3: type = converted %.loc7_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Y.loc7_15.2: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y.loc7_15.1 (constants.%Y)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.I.impl.%C.loc7_24.2, @I [symbolic = @C.as.I.impl.%.loc7_31 (constants.%.243)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .F = imports.%Main.import_ref.046
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
// CHECK:STDOUT:   %C.loc7_24.1: type = class_type @C, @C(%Y.loc7_15.1) [symbolic = %C.loc7_24.1 (constants.%C.685960.2)]
// CHECK:STDOUT:   %.loc7_31: <witness> = impl_self_witness %C.loc7_24.1, @I [symbolic = %.loc7_31 (constants.%.243)]
// CHECK:STDOUT:   %I.impl_witness.loc7_31.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %I.impl_witness.loc7_31.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.1: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %C.as.I.impl.F.type.loc8_24.1 (constants.%C.as.I.impl.F.type.95b168.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.1: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.1 (%C.as.I.impl.F.type.95b168.1) = struct_value () [symbolic = %C.as.I.impl.F.loc8_24.1 (constants.%C.as.I.impl.F.bf08b1.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.2: type = fn_type @C.as.I.impl.F.loc8_24.2, @C.as.I.impl(%Y.loc7_15.1) [symbolic = %C.as.I.impl.F.type.loc8_24.2 (constants.%C.as.I.impl.F.type.95b168.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.2: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.2 (%C.as.I.impl.F.type.95b168.2) = struct_value () [symbolic = %C.as.I.impl.F.loc8_24.2 (constants.%C.as.I.impl.F.bf08b1.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc7_24.2 as %I.ref {
// CHECK:STDOUT:     %C.as.I.impl.F.decl.loc8_24.1: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.1 (%C.as.I.impl.F.type.95b168.1) = fn_decl @C.as.I.impl.F.loc8_24.1 [symbolic = @C.as.I.impl.%C.as.I.impl.F.loc8_24.1 (constants.%C.as.I.impl.F.bf08b1.1)] {
// CHECK:STDOUT:       %x.param_patt.loc8_16.1: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.a22) = value_param_pattern [symbolic = %x.param_patt.loc8_16.2 (constants.%x.param_patt.7af)]
// CHECK:STDOUT:       %x.patt.loc8_16.1: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.a22) = wrapper_binding_pattern x, %x.param_patt.loc8_16.1 [symbolic = %x.patt.loc8_16.2 (constants.%x.patt.574)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.685960.2) = value_param call_param0
// CHECK:STDOUT:       %.loc8: type = splice_block %C.loc8_21.2 [symbolic = %C.loc8_21.1 (constants.%C.685960.2)] {
// CHECK:STDOUT:         %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:         %Y.ref: %empty_tuple.type = name_ref Y, @C.as.I.impl.%Y.loc7_15.2 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:         %C.loc8_21.2: type = class_type @C, @C(constants.%Y) [symbolic = %C.loc8_21.1 (constants.%C.685960.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.685960.2) = wrapper_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.as.I.impl.F.decl.loc8_24.2: @C.as.I.impl.%C.as.I.impl.F.type.loc8_24.2 (%C.as.I.impl.F.type.95b168.2) = fn_decl @C.as.I.impl.F.loc8_24.2 [symbolic = @C.as.I.impl.%C.as.I.impl.F.loc8_24.2 (constants.%C.as.I.impl.F.bf08b1.2)] {
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
// CHECK:STDOUT:     .Self = constants.%C.685960.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.e59c42.2: %I.type) [from "a.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.loc8_24.1(@C.as.I.impl.%Y.loc7_15.2: %empty_tuple.type) {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.loc8_21.1: type = class_type @C, @C(%Y) [symbolic = %C.loc8_21.1 (constants.%C.685960.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc8_21.1 [symbolic = %pattern_type (constants.%pattern_type.a22)]
// CHECK:STDOUT:   %x.param_patt.loc8_16.2: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.a22) = value_param_pattern [symbolic = %x.param_patt.loc8_16.2 (constants.%x.param_patt.7af)]
// CHECK:STDOUT:   %x.patt.loc8_16.2: @C.as.I.impl.F.loc8_24.1.%pattern_type (%pattern_type.a22) = wrapper_binding_pattern x, %x.param_patt.loc8_16.2 [symbolic = %x.patt.loc8_16.2 (constants.%x.patt.574)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc8_21.1 [symbolic = %require_complete (constants.%require_complete.544)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @C.as.I.impl.F.loc8_24.1.%C.loc8_21.1 (%C.685960.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.loc8_24.2(@C.as.I.impl.%Y.loc7_15.2: %empty_tuple.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F.loc8_24.1, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.95b168.1)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.95b168.1) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.bf08b1.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.loc8_24.2: <specific function> = specific_function %C.as.I.impl.F, @C.as.I.impl.F.loc8_24.1(%Y) [symbolic = %C.as.I.impl.F.specific_fn.loc8_24.2 (constants.%C.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.685960.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete.544)]
// CHECK:STDOUT:   %C.val: @C.as.I.impl.F.loc8_24.2.%C (%C.685960.2) = struct_value () [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:   %.9: ref @C.as.I.impl.F.loc8_24.2.%C (%C.685960.2) = temporary invalid, %C.val [symbolic = %.9 (constants.%.cba)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.490)]
// CHECK:STDOUT:   %Destroy.facet.3: %Destroy.type = facet_value %C, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.bf5)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.a37)]
// CHECK:STDOUT:   %.10: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.3 [symbolic = %.10 (constants.%.513)]
// CHECK:STDOUT:   %impl.elem2.2: @C.as.I.impl.F.loc8_24.2.%.10 (%.513) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %.9, %impl.elem2.2 [symbolic = %bound_method.3 (constants.%bound_method.f09)]
// CHECK:STDOUT:   %specific_impl_fn.2: <specific function> = specific_impl_function %impl.elem2.2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.3) [symbolic = %specific_impl_fn.2 (constants.%specific_impl_fn.855)]
// CHECK:STDOUT:   %bound_method.4: <bound method> = bound_method %.9, %specific_impl_fn.2 [symbolic = %bound_method.4 (constants.%bound_method.6af)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: %empty_struct_type) [thunk @C.as.I.impl.%C.as.I.impl.F.decl.loc8_24.1 for imports.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet)] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.95b168.1) = specific_constant @C.as.I.impl.%C.as.I.impl.F.decl.loc8_24.1, @C.as.I.impl(constants.%Y) [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.bf08b1.1)]
// CHECK:STDOUT:     %F.ref: @C.as.I.impl.F.loc8_24.2.%C.as.I.impl.F.type (%C.as.I.impl.F.type.95b168.1) = name_ref F, %.loc8 [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.bf08b1.1)]
// CHECK:STDOUT:     %C.as.I.impl.F.specific_fn.loc8_24.1: <specific function> = specific_function %F.ref, @C.as.I.impl.F.loc8_24.1(constants.%Y) [symbolic = %C.as.I.impl.F.specific_fn.loc8_24.2 (constants.%C.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %.2: ref @C.as.I.impl.F.loc8_24.2.%C (%C.685960.2) = temporary_storage
// CHECK:STDOUT:     %.3: init @C.as.I.impl.F.loc8_24.2.%C (%C.685960.2) to %.2 = class_init () [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %.4: init @C.as.I.impl.F.loc8_24.2.%C (%C.685960.2) = converted %x.param, %.3 [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %.5: ref @C.as.I.impl.F.loc8_24.2.%C (%C.685960.2) = temporary %.2, %.4 [symbolic = %.9 (constants.%.cba)]
// CHECK:STDOUT:     %.6: @C.as.I.impl.F.loc8_24.2.%C (%C.685960.2) = acquire_value %.5 [symbolic = %C.val (constants.%C.val)]
// CHECK:STDOUT:     %C.as.I.impl.F.call: init %empty_tuple.type = call %C.as.I.impl.F.specific_fn.loc8_24.1(%.6)
// CHECK:STDOUT:     %SelfDestruct.ref: %Destroy.assoc_type = name_ref SelfDestruct, imports.%Core.import_ref.e39 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.1: @C.as.I.impl.F.loc8_24.2.%.10 (%.513) = impl_witness_access constants.%Destroy.lookup_impl_witness.490, element2 [symbolic = %impl.elem2.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.1: <bound method> = bound_method %.5, %impl.elem2.1 [symbolic = %bound_method.3 (constants.%bound_method.f09)]
// CHECK:STDOUT:     %Destroy.facet.1: %Destroy.type = facet_value constants.%C.685960.2, (constants.%Destroy.lookup_impl_witness.490) [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.bf5)]
// CHECK:STDOUT:     %.7: %Destroy.type = converted constants.%C.685960.2, %Destroy.facet.1 [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.bf5)]
// CHECK:STDOUT:     %Destroy.facet.2: %Destroy.type = facet_value constants.%C.685960.2, (constants.%Destroy.lookup_impl_witness.490) [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.bf5)]
// CHECK:STDOUT:     %.8: %Destroy.type = converted constants.%C.685960.2, %Destroy.facet.2 [symbolic = %Destroy.facet.3 (constants.%Destroy.facet.bf5)]
// CHECK:STDOUT:     %specific_impl_fn.1: <specific function> = specific_impl_function %impl.elem2.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Destroy.facet.bf5) [symbolic = %specific_impl_fn.2 (constants.%specific_impl_fn.855)]
// CHECK:STDOUT:     %bound_method.2: <bound method> = bound_method %.5, %specific_impl_fn.1 [symbolic = %bound_method.4 (constants.%bound_method.6af)]
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
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.418
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ca5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.dda) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%Y) {
// CHECK:STDOUT:   %Y.patt.loc7_15.2 => constants.%Y.patt
// CHECK:STDOUT:   %Y.loc7_15.1 => constants.%Y
// CHECK:STDOUT:   %C.loc7_24.1 => constants.%C.685960.2
// CHECK:STDOUT:   %.loc7_31 => constants.%.243
// CHECK:STDOUT:   %I.impl_witness.loc7_31.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.1 => constants.%C.as.I.impl.F.type.95b168.1
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.1 => constants.%C.as.I.impl.F.bf08b1.1
// CHECK:STDOUT:   %C.as.I.impl.F.type.loc8_24.2 => constants.%C.as.I.impl.F.type.95b168.2
// CHECK:STDOUT:   %C.as.I.impl.F.loc8_24.2 => constants.%C.as.I.impl.F.bf08b1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.loc8_24.1(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C.loc8_21.1 => constants.%C.685960.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a22
// CHECK:STDOUT:   %x.param_patt.loc8_16.2 => constants.%x.param_patt.7af
// CHECK:STDOUT:   %x.patt.loc8_16.2 => constants.%x.patt.574
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.544
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.7c0
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.11d
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
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: %empty_tuple.type = symbolic_binding X, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %X.patt: %pattern_type.cb1 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.bf2: type = class_type @C, @C(%empty_tuple) [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.dda: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.418: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.dda) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.ca5: %I.WithSelf.F.type.418 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.9f7 [concrete]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic]
// CHECK:STDOUT:   %C.685960.2: type = class_type @C, @C(%Y) [symbolic]
// CHECK:STDOUT:   %.243: <witness> = impl_self_witness %C.685960.2, @I [symbolic]
// CHECK:STDOUT:   %I.impl_witness.735: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.3084a6.1: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.2c864c.1: %C.as.I.impl.F.type.3084a6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.type.3084a6.2: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.2c864c.2: %C.as.I.impl.F.type.3084a6.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a22: type = pattern_type %C.685960.2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.7af: %pattern_type.a22 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [symbolic]
// CHECK:STDOUT:   %x.patt.574: %pattern_type.a22 = wrapper_binding_pattern x, %x.param_patt.7af [symbolic]
// CHECK:STDOUT:   %require_complete.544: <witness> = require_complete_type %C.685960.2 [symbolic]
// CHECK:STDOUT:   %I.facet.bf9: %I.type = facet_value %C.685960.2, (%I.impl_witness.735) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.490: <witness> = lookup_impl_witness %C.685960.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.bf5: %Destroy.type = facet_value %C.685960.2, (%Destroy.lookup_impl_witness.490) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.a37: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.bf5) [symbolic]
// CHECK:STDOUT:   %.513: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.a37, %Destroy.facet.bf5 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.513 = impl_witness_access %Destroy.lookup_impl_witness.490, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.855: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.bf5) [symbolic]
// CHECK:STDOUT:   %C.val.a97: %C.685960.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.cba: ref %C.685960.2 = temporary invalid, %C.val.a97 [symbolic]
// CHECK:STDOUT:   %bound_method.6af: <bound method> = bound_method %.cba, %specific_impl_fn.855 [symbolic]
// CHECK:STDOUT:   %bound_method.f09: <bound method> = bound_method %.cba, %impl.elem2 [symbolic]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.bd3: <specific function> = specific_function %C.as.I.impl.F.2c864c.2, @C.as.I.impl.F.2(%Y) [symbolic]
// CHECK:STDOUT:   %.cc4: <witness> = impl_self_witness %C.bf2, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.141: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.680f40.1: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.c2d40b.1: %C.as.I.impl.F.type.680f40.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.680f40.2: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.c2d40b.2: %C.as.I.impl.F.type.680f40.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.a3e: %I.type = facet_value %C.bf2, (%I.impl_witness.141) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.85c: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.a3e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.13f: %I.WithSelf.F.type.85c = struct_value () [concrete]
// CHECK:STDOUT:   %.75b: type = fn_type_with_self_type %I.WithSelf.F.type.85c, %I.facet.a3e [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.74f6fe.1: <specific function> = specific_function %C.as.I.impl.F.c2d40b.2, @C.as.I.impl.F.1(%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type.781: type = pattern_type %C.bf2 [concrete]
// CHECK:STDOUT:   %x.param_patt.d18: %pattern_type.781 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.12f: %pattern_type.781 = wrapper_binding_pattern x, %x.param_patt.d18 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn.74f6fe.2: <specific function> = specific_function %C.as.I.impl.F.c2d40b.1, @C.as.I.impl.F.2(%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.val.f21: %C.bf2 = struct_value () [concrete]
// CHECK:STDOUT:   %.8d2: ref %C.bf2 = temporary invalid, %C.val.f21 [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc7_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.fef: %pattern_type.781 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5be: %pattern_type.781 = wrapper_binding_pattern self, %self.param_patt.fef [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc7_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc7_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %Destroy.WithSelf.SubobjectDestroy.d01daf.2, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.fb3: %Destroy.type = facet_value %C.bf2, (%custom_witness.f8f19d.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.926: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.fb3) [concrete]
// CHECK:STDOUT:   %.045: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.926, %Destroy.facet.fb3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.8d2, %Destroy.WithSelf.SelfDestruct.db3fdb.2 [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.172 = import_ref Main//b, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c90e1a.1: %empty_tuple.type = import_ref Main//b, loc5_10, loaded [symbolic = @C.%X (constants.%X)]
// CHECK:STDOUT:   %Main.import_ref.1b3: %I.assoc_type = import_ref Main//a, loc5_14, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.F.006 = import_ref Main//a, F, unloaded
// CHECK:STDOUT:   %I.WithSelf.F.decl: %I.WithSelf.F.type.418 = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ca5)] {} {}
// CHECK:STDOUT:   %Main.import_ref.e59c42.2: %I.type = import_ref Main//a, loc4_13, loaded [symbolic = constants.%Self.dda]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//a, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.9f7: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.418) = import_ref Main//a, loc5_14, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ca5)]
// CHECK:STDOUT:   %Main.import_ref.aa5: <witness> = import_ref Main//b, loc7_31, loaded [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness.735)]
// CHECK:STDOUT:   %Main.import_ref.421: @C.as.I.impl.%C.as.I.impl.F.type.2 (%C.as.I.impl.F.type.3084a6.1) = import_ref Main//b, loc8_24, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F.2 (constants.%C.as.I.impl.F.2c864c.1)]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.421), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.2: %empty_tuple.type = import_ref Main//b, loc7_15, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.F.b5b: @C.as.I.impl.%C.as.I.impl.F.type.1 (%C.as.I.impl.F.type.3084a6.2) = import_ref Main//b, F, loaded [symbolic = @C.as.I.impl.%C.as.I.impl.F.1 (constants.%C.as.I.impl.F.2c864c.2)]
// CHECK:STDOUT:   %Main.import_ref.c90e1a.3: %empty_tuple.type = import_ref Main//b, loc7_15, loaded [symbolic = @C.as.I.impl.%Y (constants.%Y)]
// CHECK:STDOUT:   %Main.import_ref.0ee: type = import_ref Main//b, loc7_24, loaded [symbolic = @C.as.I.impl.%C (constants.%C.685960.2)]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.781 = ref_param_pattern [concrete = constants.%self.param_patt.fef]
// CHECK:STDOUT:     %self.patt: %pattern_type.781 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5be]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.bf2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.bf2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc7_16.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.781 = ref_param_pattern [concrete = constants.%self.param_patt.fef]
// CHECK:STDOUT:     %self.patt: %pattern_type.781 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5be]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.bf2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.bf2 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   .F = imports.%Main.import_ref.1b3
// CHECK:STDOUT:   witness = (imports.%Main.F.006)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(imports.%Main.import_ref.c90e1a.4: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y.patt: %pattern_type.cb1 = symbolic_binding_pattern Y, 0 [symbolic = %Y.patt (constants.%Y.patt)]
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.685960.2)]
// CHECK:STDOUT:   %.1: <witness> = impl_self_witness %C, @I [symbolic = %.1 (constants.%.243)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl(%Y) [symbolic = %I.impl_witness (constants.%I.impl_witness.735)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type.1 (constants.%C.as.I.impl.F.type.3084a6.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.1: @C.as.I.impl.%C.as.I.impl.F.type.1 (%C.as.I.impl.F.type.3084a6.2) = struct_value () [symbolic = %C.as.I.impl.F.1 (constants.%C.as.I.impl.F.2c864c.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.type.2: type = fn_type @C.as.I.impl.F.1, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type.2 (constants.%C.as.I.impl.F.type.3084a6.1)]
// CHECK:STDOUT:   %C.as.I.impl.F.2: @C.as.I.impl.%C.as.I.impl.F.type.2 (%C.as.I.impl.F.type.3084a6.1) = struct_value () [symbolic = %C.as.I.impl.F.2 (constants.%C.as.I.impl.F.2c864c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.0ee as imports.%Main.import_ref.e90 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.aa5
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
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.172
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %.loc7_6: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_7: %empty_tuple.type = converted %.loc7_6, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%empty_tuple) [concrete = constants.%C.bf2]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %F.ref.loc7_11: %I.assoc_type = name_ref F, imports.%Main.import_ref.1b3 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (constants.%I.impl_witness.141) [concrete = constants.%I.facet.a3e]
// CHECK:STDOUT:   %.loc7_8: %I.type = converted %C, %I.facet [concrete = constants.%I.facet.a3e]
// CHECK:STDOUT:   %impl.elem0: %.75b = impl_witness_access constants.%I.impl_witness.141, element0 [concrete = constants.%C.as.I.impl.F.c2d40b.2]
// CHECK:STDOUT:   %.loc7_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.I.impl.F.1(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.specific_fn.74f6fe.1]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_16.2: %empty_struct_type = converted %.loc7_16.1, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_17: %C.as.I.impl.F.type.680f40.1 = specific_constant imports.%Main.F.b5b, @C.as.I.impl(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.c2d40b.1]
// CHECK:STDOUT:   %F.ref.loc7_17: %C.as.I.impl.F.type.680f40.1 = name_ref F, %.loc7_17 [concrete = constants.%C.as.I.impl.F.c2d40b.1]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %F.ref.loc7_17, @C.as.I.impl.F.2(constants.%empty_tuple) [concrete = constants.%C.as.I.impl.F.specific_fn.74f6fe.2]
// CHECK:STDOUT:   %.loc7_16.3: ref %C.bf2 = temporary_storage
// CHECK:STDOUT:   %.loc7_16.4: init %C.bf2 to %.loc7_16.3 = class_init () [concrete = constants.%C.val.f21]
// CHECK:STDOUT:   %.loc7_16.5: init %C.bf2 = converted %.loc7_16.2, %.loc7_16.4 [concrete = constants.%C.val.f21]
// CHECK:STDOUT:   %.loc7_16.6: ref %C.bf2 = temporary %.loc7_16.3, %.loc7_16.5 [concrete = constants.%.8d2]
// CHECK:STDOUT:   %.loc7_16.7: %C.bf2 = acquire_value %.loc7_16.6 [concrete = constants.%C.val.f21]
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %C.as.I.impl.F.specific_fn(%.loc7_16.7)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.8d2)
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
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F.2, @C.as.I.impl(%Y) [symbolic = %C.as.I.impl.F.type (constants.%C.as.I.impl.F.type.3084a6.2)]
// CHECK:STDOUT:   %C.as.I.impl.F: @C.as.I.impl.F.1.%C.as.I.impl.F.type (%C.as.I.impl.F.type.3084a6.2) = struct_value () [symbolic = %C.as.I.impl.F (constants.%C.as.I.impl.F.2c864c.2)]
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn: <specific function> = specific_function %C.as.I.impl.F, @C.as.I.impl.F.2(%Y) [symbolic = %C.as.I.impl.F.specific_fn (constants.%C.as.I.impl.F.specific_fn.bd3)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.685960.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete.544)]
// CHECK:STDOUT:   %C.val: @C.as.I.impl.F.1.%C (%C.685960.2) = struct_value () [symbolic = %C.val (constants.%C.val.a97)]
// CHECK:STDOUT:   %.1: ref @C.as.I.impl.F.1.%C (%C.685960.2) = temporary invalid, %C.val [symbolic = %.1 (constants.%.cba)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.490)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.bf5)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.a37)]
// CHECK:STDOUT:   %.2: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet [symbolic = %.2 (constants.%.513)]
// CHECK:STDOUT:   %impl.elem2: @C.as.I.impl.F.1.%.2 (%.513) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.1, %impl.elem2 [symbolic = %bound_method.1 (constants.%bound_method.f09)]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet) [symbolic = %specific_impl_fn (constants.%specific_impl_fn.855)]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.1, %specific_impl_fn [symbolic = %bound_method.2 (constants.%bound_method.6af)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn [thunk imports.%Main.F.b5b for imports.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.bf9)];
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.I.impl.F.2(imports.%Main.import_ref.c90e1a.2: %empty_tuple.type) [from "b.carbon"] {
// CHECK:STDOUT:   %Y: %empty_tuple.type = symbolic_binding Y, 0 [symbolic = %Y (constants.%Y)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%Y) [symbolic = %C (constants.%C.685960.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.a22)]
// CHECK:STDOUT:   %x.param_patt: @C.as.I.impl.F.2.%pattern_type (%pattern_type.a22) = value_param_pattern [symbolic = %x.param_patt (constants.%x.param_patt.7af)]
// CHECK:STDOUT:   %x.patt: @C.as.I.impl.F.2.%pattern_type (%pattern_type.a22) = wrapper_binding_pattern x, %x.param_patt [symbolic = %x.patt (constants.%x.patt.574)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C [symbolic = %require_complete (constants.%require_complete.544)]
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_16.2(%self.param: ref %C.bf2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_16.2(%self.param: ref %C.bf2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_16.2(%self.param: ref %C.bf2) {
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
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.418
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ca5
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
// CHECK:STDOUT:   %C => constants.%C.685960.2
// CHECK:STDOUT:   %.1 => constants.%.243
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.735
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1 => constants.%C.as.I.impl.F.type.3084a6.2
// CHECK:STDOUT:   %C.as.I.impl.F.1 => constants.%C.as.I.impl.F.2c864c.2
// CHECK:STDOUT:   %C.as.I.impl.F.type.2 => constants.%C.as.I.impl.F.type.3084a6.1
// CHECK:STDOUT:   %C.as.I.impl.F.2 => constants.%C.as.I.impl.F.2c864c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.2(constants.%Y) {
// CHECK:STDOUT:   %Y => constants.%Y
// CHECK:STDOUT:   %C => constants.%C.685960.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a22
// CHECK:STDOUT:   %x.param_patt => constants.%x.param_patt.7af
// CHECK:STDOUT:   %x.patt => constants.%x.patt.574
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.544
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.bf9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.1(constants.%Y) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y.patt => constants.%Y.patt
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.bf2
// CHECK:STDOUT:   %.1 => constants.%.cc4
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.141
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.I.impl.F.type.1 => constants.%C.as.I.impl.F.type.680f40.1
// CHECK:STDOUT:   %C.as.I.impl.F.1 => constants.%C.as.I.impl.F.c2d40b.1
// CHECK:STDOUT:   %C.as.I.impl.F.type.2 => constants.%C.as.I.impl.F.type.680f40.2
// CHECK:STDOUT:   %C.as.I.impl.F.2 => constants.%C.as.I.impl.F.c2d40b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.a3e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.a3e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.85c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.13f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.1(constants.%empty_tuple) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C.as.I.impl.F.type => constants.%C.as.I.impl.F.type.680f40.1
// CHECK:STDOUT:   %C.as.I.impl.F => constants.%C.as.I.impl.F.c2d40b.1
// CHECK:STDOUT:   %C.as.I.impl.F.specific_fn => constants.%C.as.I.impl.F.specific_fn.74f6fe.2
// CHECK:STDOUT:   %C => constants.%C.bf2
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %C.val => constants.%C.val.f21
// CHECK:STDOUT:   %.1 => constants.%.8d2
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.2
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.fb3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.926
// CHECK:STDOUT:   %.2 => constants.%.045
// CHECK:STDOUT:   %impl.elem2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %bound_method.1 => constants.%Destroy.WithSelf.SelfDestruct.bound
// CHECK:STDOUT:   %specific_impl_fn => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %bound_method.2 => constants.%Destroy.WithSelf.SelfDestruct.bound
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.F.2(constants.%empty_tuple) {
// CHECK:STDOUT:   %Y => constants.%empty_tuple
// CHECK:STDOUT:   %C => constants.%C.bf2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.781
// CHECK:STDOUT:   %x.param_patt => constants.%x.param_patt.d18
// CHECK:STDOUT:   %x.patt => constants.%x.patt.12f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
