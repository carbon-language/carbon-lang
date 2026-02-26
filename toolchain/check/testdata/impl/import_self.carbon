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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_self.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

class C { adapt (); }

impl C as Add {
  fn Op[unused self: Self](unused other: Self) -> Self { return () as C; }
}

fn F(x: C, y: C) -> C {
  return x.(Add.Op)(y);
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.0ec: Core.Form = init_form %Self.binding.as_type, call_param2 [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.type: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op: %Add.WithSelf.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type @Add [concrete]
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, @Add.WithSelf.%Add.WithSelf.Op.decl [concrete]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [concrete = constants.%Add.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Add.WithSelf.decl = interface_with_self_decl @Add [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Add.WithSelf.Op.decl: @Add.WithSelf.%Add.WithSelf.Op.type (%Add.WithSelf.Op.type) = fn_decl @Add.WithSelf.Op [symbolic = @Add.WithSelf.%Add.WithSelf.Op (constants.%Add.WithSelf.Op)] {
// CHECK:STDOUT:     %self.patt: @Add.WithSelf.Op.%pattern_type (%pattern_type) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Add.WithSelf.Op.%pattern_type (%pattern_type) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:     %other.patt: @Add.WithSelf.Op.%pattern_type (%pattern_type) = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @Add.WithSelf.Op.%pattern_type (%pattern_type) = value_param_pattern %other.patt [concrete]
// CHECK:STDOUT:     %return.patt: @Add.WithSelf.Op.%pattern_type (%pattern_type) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Add.WithSelf.Op.%pattern_type (%pattern_type) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_37: type = facet_access_type %Self.ref.loc5_37 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc5_37.2: type = converted %Self.ref.loc5_37, %Self.as_type.loc5_37 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc5_37.3: Core.Form = init_form %.loc5_37.2, call_param2 [symbolic = %.loc5_37.1 (constants.%.0ec)]
// CHECK:STDOUT:     %self.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_15.1: type = splice_block %.loc5_15.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_15: type = facet_access_type %Self.ref.loc5_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_15.2: type = converted %Self.ref.loc5_15, %Self.as_type.loc5_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc5_28.1: type = splice_block %.loc5_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_28: type = facet_access_type %Self.ref.loc5_28 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc5_28.2: type = converted %Self.ref.loc5_28, %Self.as_type.loc5_28 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, %Add.WithSelf.Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = @Add.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Add.WithSelf.%Add.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Add.WithSelf.Op(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:   %.loc5_37.1: Core.Form = init_form %Self.binding.as_type, call_param2 [symbolic = %.loc5_37.1 (constants.%.0ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type), %other.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type)) -> out %return.param: @Add.WithSelf.Op.%Self.binding.as_type (%Self.binding.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT:   %.loc5_37.1 => constants.%.0ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple.af4: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.type.099: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Add.WithSelf.Op.99a: %Add.WithSelf.Op.type.099 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.5af: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.14a: Core.Form = init_form %Self.binding.as_type, call_param2 [symbolic]
// CHECK:STDOUT:   %Add.impl_witness: <witness> = impl_witness @C.as.Add.impl.%Add.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.92f: Core.Form = init_form %C, call_param2 [concrete]
// CHECK:STDOUT:   %C.as.Add.impl.Op.type: type = fn_type @C.as.Add.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Add.impl.Op: %C.as.Add.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %C, (%Add.impl_witness) [concrete]
// CHECK:STDOUT:   %Add.WithSelf.Op.type.3ed: type = fn_type @Add.WithSelf.Op, @Add.WithSelf(%Add.facet) [concrete]
// CHECK:STDOUT:   %Add.WithSelf.Op.5d9: %Add.WithSelf.Op.type.3ed = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.bfb: %C = tuple_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type @Add [concrete]
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, imports.%Main.import_ref.dad [concrete]
// CHECK:STDOUT:   %.c75: type = fn_type_with_self_type %Add.WithSelf.Op.type.3ed, %Add.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Add: type = import_ref Main//a, Add, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.a96: %Add.assoc_type = import_ref Main//a, loc5_41, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.Op: @Add.WithSelf.%Add.WithSelf.Op.type (%Add.WithSelf.Op.type.099) = import_ref Main//a, Op, loaded [symbolic = @Add.WithSelf.%Add.WithSelf.Op (constants.%Add.WithSelf.Op.99a)]
// CHECK:STDOUT:   %Main.import_ref.bb084d.2: %Add.type = import_ref Main//a, loc4_15, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.17c = import_ref Main//a, loc4_15, unloaded
// CHECK:STDOUT:   %Main.import_ref.dad: @Add.WithSelf.%Add.WithSelf.Op.type (%Add.WithSelf.Op.type.099) = import_ref Main//a, loc5_41, loaded [symbolic = @Add.WithSelf.%Add.WithSelf.Op (constants.%Add.WithSelf.Op.99a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Add = imports.%Main.Add
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Add.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%Main.Add [concrete = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.7c7 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7c7 = value_param_pattern %x.patt [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.7c7 = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.7c7 = value_param_pattern %y.patt [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c7 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c7 = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc12_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %C.ref.loc12_21, call_param2 [concrete = constants.%.92f]
// CHECK:STDOUT:     %x.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc12_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc12_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %y: %C = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.17c
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.a96
// CHECK:STDOUT:   witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Add.impl: %C.ref as %Add.ref {
// CHECK:STDOUT:   %C.as.Add.impl.Op.decl: %C.as.Add.impl.Op.type = fn_decl @C.as.Add.impl.Op [concrete = constants.%C.as.Add.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.7c7 = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.7c7 = value_param_pattern %other.patt [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c7 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c7 = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc9_51: type = name_ref Self, @C.as.Add.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %.loc9_51: Core.Form = init_form %Self.ref.loc9_51, call_param2 [concrete = constants.%.92f]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc9_22: type = name_ref Self, @C.as.Add.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc9_42: type = name_ref Self, @C.as.Add.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.impl_witness_table = impl_witness_table (%C.as.Add.impl.Op.decl), @C.as.Add.impl [concrete]
// CHECK:STDOUT:   %Add.impl_witness: <witness> = impl_witness %Add.impl_witness_table [concrete = constants.%Add.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Add.impl.Op.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   witness = %Add.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc6_18: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc6_19: type = converted %.loc6_18, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   adapt_decl %.loc6_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_tuple.type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Add.WithSelf.Op(imports.%Main.import_ref.bb084d.2: %Add.type) [from "a.carbon"] {
// CHECK:STDOUT:   %Self: %Add.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.5af)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %Self.binding.as_type, call_param2 [symbolic = %.1 (constants.%.14a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Add.impl.Op(%self.param: %C, %other.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_66: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc9_68.1: %C = as_compatible %empty_tuple [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc9_68.2: %C = converted %.loc9_66, %.loc9_68.1 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc9_72.1: %empty_tuple.type = as_compatible %.loc9_68.2 [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc9_72.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc9_72.3: init %C = as_compatible %.loc9_72.2 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc9_72.4: init %C = converted %.loc9_68.2, %.loc9_72.3 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   return %.loc9_72.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %C, %y.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %Add.ref: type = name_ref Add, imports.%Main.Add [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Op.ref: %Add.assoc_type = name_ref Op, imports.%Main.import_ref.a96 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.c75 = impl_witness_access constants.%Add.impl_witness, element0 [concrete = constants.%C.as.Add.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %y.ref: %C = name_ref y, %y
// CHECK:STDOUT:   %C.as.Add.impl.Op.call: init %C = call %bound_method(%x.ref, %y.ref)
// CHECK:STDOUT:   return %C.as.Add.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Add.WithSelf.Op.type => constants.%Add.WithSelf.Op.type.099
// CHECK:STDOUT:   %Add.WithSelf.Op => constants.%Add.WithSelf.Op.99a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5af
// CHECK:STDOUT:   %.1 => constants.%.14a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf(constants.%Add.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Add.facet
// CHECK:STDOUT:   %Add.WithSelf.Op.type => constants.%Add.WithSelf.Op.type.3ed
// CHECK:STDOUT:   %Add.WithSelf.Op => constants.%Add.WithSelf.Op.5d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.WithSelf.Op(constants.%Add.facet) {
// CHECK:STDOUT:   %Self => constants.%Add.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7c7
// CHECK:STDOUT:   %.1 => constants.%.92f
// CHECK:STDOUT: }
// CHECK:STDOUT:
