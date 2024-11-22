// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_self.carbon

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
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [template]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %Add.type, %Op.type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @Add.%Op.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [template = constants.%Add.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: %Op.type = fn_decl @Op [template = constants.%Op] {
// CHECK:STDOUT:     %self.patt: @Op.%.loc5_15.1 (%.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.%.loc5_15.1 (%.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @Op.%.loc5_15.1 (%.1) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.%.loc5_15.1 (%.1) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @Op.%.loc5_15.1 (%.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.%.loc5_15.1 (%.1) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_15.2: type = facet_access_type %Self.ref.loc5_15 [symbolic = %.loc5_15.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc5_15.3: type = converted %Self.ref.loc5_15, %.loc5_15.2 [symbolic = %.loc5_15.1 (constants.%.1)]
// CHECK:STDOUT:     %Self.ref.loc5_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_28.1: type = facet_access_type %Self.ref.loc5_28 [symbolic = %.loc5_15.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc5_28.2: type = converted %Self.ref.loc5_28, %.loc5_28.1 [symbolic = %.loc5_15.1 (constants.%.1)]
// CHECK:STDOUT:     %Self.ref.loc5_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_37.1: type = facet_access_type %Self.ref.loc5_37 [symbolic = %.loc5_15.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc5_37.2: type = converted %Self.ref.loc5_37, %.loc5_37.1 [symbolic = %.loc5_15.1 (constants.%.1)]
// CHECK:STDOUT:     %self.param: @Op.%.loc5_15.1 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @Op.%.loc5_15.1 (%.1) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.%.loc5_15.1 (%.1) = value_param runtime_param1
// CHECK:STDOUT:     %other: @Op.%.loc5_15.1 (%.1) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.%.loc5_15.1 (%.1) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @Op.%.loc5_15.1 (%.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5: %.2 = assoc_entity element0, %Op.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %.loc5_15.1: type = facet_access_type %Self [symbolic = %.loc5_15.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.%.loc5_15.1 (%.1)](%other.param_patt: @Op.%.loc5_15.1 (%.1)) -> @Op.%.loc5_15.1 (%.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %.loc5_15.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %.2: %Add.type = facet_value %empty_tuple.type, %empty_tuple.type [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %Add.type, %Op.type.2 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//a, inst+1, loaded [template = constants.%Add.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//a, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: %.4 = import_ref Main//a, inst+34, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.4: %Op.type.2 = import_ref Main//a, inst+27, loaded [template = constants.%Op.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Add = imports.%import_ref.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%import_ref.1 [template = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %y.patt: %empty_tuple.type = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: %empty_tuple.type = value_param_pattern %y.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_10.2: type = converted %.loc10_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_17.2: type = converted %.loc10_17.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_24.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_24.2: type = converted %.loc10_24.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:     %y.param: %empty_tuple.type = value_param runtime_param1
// CHECK:STDOUT:     %y: %empty_tuple.type = bind_name y, %y.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .Op = imports.%import_ref.3
// CHECK:STDOUT:   witness = (imports.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc6_7.2 as %Add.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %empty_tuple.type = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %empty_tuple.type = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc7_15: type = name_ref Self, @impl.%.loc6_7.2 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Self.ref.loc7_28: type = name_ref Self, @impl.%.loc6_7.2 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Self.ref.loc7_37: type = name_ref Self, @impl.%.loc6_7.2 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %empty_tuple.type = value_param runtime_param1
// CHECK:STDOUT:     %other: %empty_tuple.type = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_16: <witness> = interface_witness (%Op.decl) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc6_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[%self.param_patt: %empty_tuple.type](%other.param_patt: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_52: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_53: %empty_tuple.type = converted %.loc7_52, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc7_53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%.1 (%.1)](%other.param_patt: @Op.2.%.1 (%.1)) -> @Op.2.%.1 (%.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %empty_tuple.type, %y.param_patt: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_tuple.type = name_ref x, %x
// CHECK:STDOUT:   %Add.ref: type = name_ref Add, imports.%import_ref.1 [template = constants.%Add.type]
// CHECK:STDOUT:   %Op.ref: %.4 = name_ref Op, imports.%import_ref.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_11.1: %Op.type.2 = interface_witness_access constants.%.3, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc11_11.2: <bound method> = bound_method %x.ref, %.loc11_11.1
// CHECK:STDOUT:   %y.ref: %empty_tuple.type = name_ref y, %y
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %.loc11_11.2(%x.ref, %y.ref)
// CHECK:STDOUT:   %.loc11_20.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc11_20.2: ref %empty_tuple.type = temporary %.loc11_20.1, %Op.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_23: %empty_tuple.type = converted %Op.call, %tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc11_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%.2) {
// CHECK:STDOUT:   %Self => constants.%.2
// CHECK:STDOUT:   %.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
