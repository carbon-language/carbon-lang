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

impl () as Add {
  fn Op[self: Self](other: Self) -> Self { return (); }
}

fn F(x: (), y: ()) -> () {
  return x.(Add.Op)(y);
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Add.Op.type: type = fn_type @Add.Op [concrete]
// CHECK:STDOUT:   %Add.Op: %Add.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type @Add [concrete]
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, @Add.%Add.Op.decl [concrete]
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
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Add.Op.decl: %Add.Op.type = fn_decl @Add.Op [concrete = constants.%Add.Op] {
// CHECK:STDOUT:     %self.patt: @Add.Op.%pattern_type (%pattern_type) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Add.Op.%pattern_type (%pattern_type) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: @Add.Op.%pattern_type (%pattern_type) = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @Add.Op.%pattern_type (%pattern_type) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Add.Op.%pattern_type (%pattern_type) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Add.Op.%pattern_type (%pattern_type) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_37: type = facet_access_type %Self.ref.loc5_37 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5_37: type = converted %Self.ref.loc5_37, %Self.as_type.loc5_37 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_15.1: type = splice_block %.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_15.2: type = facet_access_type %Self.ref.loc5_15 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_15.2: type = converted %Self.ref.loc5_15, %Self.as_type.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc5_28.1: type = splice_block %.loc5_28.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_28: type = facet_access_type %Self.ref.loc5_28 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_28.2: type = converted %Self.ref.loc5_28, %Self.as_type.loc5_28 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Add.assoc_type = assoc_entity element0, %Add.Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Add.Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Add.Op(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_15.1 [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type), %other.param: @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type)) -> @Add.Op.%Self.as_type.loc5_15.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Self.365: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Add.Op.type: type = fn_type @Add.Op [concrete]
// CHECK:STDOUT:   %Add.Op: %Add.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.3c7: type = facet_access_type %Self.365 [symbolic]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %Self.as_type.3c7 [symbolic]
// CHECK:STDOUT:   %Add.impl_witness: <witness> = impl_witness file.%Add.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Add.impl.Op.type: type = fn_type @empty_tuple.type.as.Add.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Add.impl.Op: %empty_tuple.type.as.Add.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %empty_tuple.type, (%Add.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Add.assoc_type: type = assoc_entity_type @Add [concrete]
// CHECK:STDOUT:   %assoc0.5ca: %Add.assoc_type = assoc_entity element0, imports.%Main.import_ref.5a3 [concrete]
// CHECK:STDOUT:   %.589: type = fn_type_with_self_type %Add.Op.type, %Add.facet [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.abd: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bae: %T.as.Destroy.impl.Op.type.abd = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.bae, @T.as.Destroy.impl.Op(%empty_tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Add: type = import_ref Main//a, Add, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.ff9 = import_ref Main//a, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f99: %Add.assoc_type = import_ref Main//a, loc5_41, loaded [concrete = constants.%assoc0.5ca]
// CHECK:STDOUT:   %Main.Op: %Add.Op.type = import_ref Main//a, Op, loaded [concrete = constants.%Add.Op]
// CHECK:STDOUT:   %Main.import_ref.2a7: %Add.type = import_ref Main//a, inst19 [no loc], loaded [symbolic = constants.%Self.365]
// CHECK:STDOUT:   %Main.import_ref.5a3: %Add.Op.type = import_ref Main//a, loc5_41, loaded [concrete = constants.%Add.Op]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Add = imports.%Main.Add
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Add.impl [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%Main.Add [concrete = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.impl_witness_table = impl_witness_table (@empty_tuple.type.as.Add.impl.%empty_tuple.type.as.Add.impl.Op.decl), @empty_tuple.type.as.Add.impl [concrete]
// CHECK:STDOUT:   %Add.impl_witness: <witness> = impl_witness %Add.impl_witness_table [concrete = constants.%Add.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.cb1 = value_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_24.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_24.2: type = converted %.loc10_24.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc10_10.1: type = splice_block %.loc10_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc10_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc10_10.3: type = converted %.loc10_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:     %y.param: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %.loc10_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc10_17.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc10_17.3: type = converted %.loc10_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %empty_tuple.type = bind_name y, %y.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add [from "a.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ff9
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Main.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Add.impl: %.loc6_7.2 as %Add.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.Add.impl.Op.decl: %empty_tuple.type.as.Add.impl.Op.type = fn_decl @empty_tuple.type.as.Add.impl.Op [concrete = constants.%empty_tuple.type.as.Add.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.cb1 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.cb1 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc7_37: type = name_ref Self, @empty_tuple.type.as.Add.impl.%.loc6_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc7_15: type = name_ref Self, @empty_tuple.type.as.Add.impl.%.loc6_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc7_28: type = name_ref Self, @empty_tuple.type.as.Add.impl.%.loc6_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %other: %empty_tuple.type = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %empty_tuple.type.as.Add.impl.Op.decl
// CHECK:STDOUT:   witness = file.%Add.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Add.Op(imports.%Main.import_ref.2a7: %Add.type) [from "a.carbon"] {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.365)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.3c7)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.2f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.Add.impl.Op(%self.param: %empty_tuple.type, %other.param: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_52: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_53: %empty_tuple.type = converted %.loc7_52, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc7_53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %empty_tuple.type, %y.param: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %Add.ref: type = name_ref Add, imports.%Main.Add [concrete = constants.%Add.type]
// CHECK:STDOUT:   %Op.ref: %Add.assoc_type = name_ref Op, imports.%Main.import_ref.f99 [concrete = constants.%assoc0.5ca]
// CHECK:STDOUT:   %impl.elem0: %.589 = impl_witness_access constants.%Add.impl_witness, element0 [concrete = constants.%empty_tuple.type.as.Add.impl.Op]
// CHECK:STDOUT:   %bound_method.loc11_11: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %y.ref: %empty_tuple.type = name_ref y, %y
// CHECK:STDOUT:   %empty_tuple.type.as.Add.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_11(%x.ref, %y.ref)
// CHECK:STDOUT:   %.loc11_22.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc11_22.2: ref %empty_tuple.type = temporary %.loc11_22.1, %empty_tuple.type.as.Add.impl.Op.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_23: %empty_tuple.type = converted %empty_tuple.type.as.Add.impl.Op.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc11_22.2, constants.%T.as.Destroy.impl.Op.bae
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.bae, @T.as.Destroy.impl.Op(constants.%empty_tuple.type) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_22: <bound method> = bound_method %.loc11_22.2, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.843 = addr_of %.loc11_22.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_22(%addr)
// CHECK:STDOUT:   return %.loc11_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.Op(constants.%Self.365) {
// CHECK:STDOUT:   %Self => constants.%Self.365
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.3c7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add.Op(constants.%Add.facet) {
// CHECK:STDOUT:   %Self => constants.%Add.facet
// CHECK:STDOUT:   %Self.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
