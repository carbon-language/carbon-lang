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
// CHECK:STDOUT:   %Add.type: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
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
// CHECK:STDOUT:     %self.patt: @Op.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:     %other.patt: @Op.%Self (%Self) = binding_pattern other
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref.loc5_15 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref.loc5_15, %.loc5_15.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %self.param: @Op.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:     %self: @Op.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:     %Self.ref.loc5_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_28.1: type = facet_type_access %Self.ref.loc5_28 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_28.2: type = converted %Self.ref.loc5_28, %.loc5_28.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %other.param: @Op.%Self (%Self) = param other, runtime_param1
// CHECK:STDOUT:     %other: @Op.%Self (%Self) = bind_name other, %other.param
// CHECK:STDOUT:     %Self.ref.loc5_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_37.1: type = facet_type_access %Self.ref.loc5_37 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_37.2: type = converted %Self.ref.loc5_37, %.loc5_37.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %return: ref @Op.%Self (%Self) = var <return slot>
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.%Self (%Self)](%other: @Op.%Self (%Self)) -> @Op.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Add.type: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %Add.type, %Op.type.2 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//a, inst+1, loaded [template = constants.%Add.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//a, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: %.3 = import_ref Main//a, inst+27, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.4: %Op.type.2 = import_ref Main//a, inst+21, loaded [template = constants.%Op.2]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//a, inst+21, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Add = imports.%import_ref.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc6_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%import_ref.1 [template = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %x.patt: %.1 = binding_pattern x
// CHECK:STDOUT:     %y.patt: %.1 = binding_pattern y
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_10.2: type = converted %.loc10_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.param: %.1 = param x, runtime_param0
// CHECK:STDOUT:     %x: %.1 = bind_name x, %x.param
// CHECK:STDOUT:     %.loc10_17.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_17.2: type = converted %.loc10_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %y.param: %.1 = param y, runtime_param1
// CHECK:STDOUT:     %y: %.1 = bind_name y, %y.param
// CHECK:STDOUT:     %.loc10_24.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_24.2: type = converted %.loc10_24.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return: ref %.1 = var <return slot>
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
// CHECK:STDOUT:     %self.patt: %.1 = binding_pattern self
// CHECK:STDOUT:     %other.patt: %.1 = binding_pattern other
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc7_15: type = name_ref Self, @impl.%.loc6_7.2 [template = constants.%.1]
// CHECK:STDOUT:     %self.param: %.1 = param self, runtime_param0
// CHECK:STDOUT:     %self: %.1 = bind_name self, %self.param
// CHECK:STDOUT:     %Self.ref.loc7_28: type = name_ref Self, @impl.%.loc6_7.2 [template = constants.%.1]
// CHECK:STDOUT:     %other.param: %.1 = param other, runtime_param1
// CHECK:STDOUT:     %other: %.1 = bind_name other, %other.param
// CHECK:STDOUT:     %Self.ref.loc7_37: type = name_ref Self, @impl.%.loc6_7.2 [template = constants.%.1]
// CHECK:STDOUT:     %return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_16: <witness> = interface_witness (%Op.decl) [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc6_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[%self: %.1](%other: %.1) -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_52: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_53: %.1 = converted %.loc7_52, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc7_53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.2.%Self (%Self)](%other: @Op.2.%Self (%Self)) -> @Op.2.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x: %.1, %y: %.1) -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.1 = name_ref x, %x
// CHECK:STDOUT:   %Add.ref: type = name_ref Add, imports.%import_ref.1 [template = constants.%Add.type]
// CHECK:STDOUT:   %Op.ref: %.3 = name_ref Op, imports.%import_ref.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_11.1: %Op.type.2 = interface_witness_access constants.%.2, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc11_11.2: <bound method> = bound_method %x.ref, %.loc11_11.1
// CHECK:STDOUT:   %y.ref: %.1 = name_ref y, %y
// CHECK:STDOUT:   %Op.call: init %.1 = call %.loc11_11.2(%x.ref, %y.ref)
// CHECK:STDOUT:   %.loc11_20.1: ref %.1 = temporary_storage
// CHECK:STDOUT:   %.loc11_20.2: ref %.1 = temporary %.loc11_20.1, %Op.call
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_23: %.1 = converted %Op.call, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc11_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%.1) {
// CHECK:STDOUT:   %Self => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
