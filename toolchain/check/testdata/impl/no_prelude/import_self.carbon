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

library "a";

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

// --- b.carbon

library "b";

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
// CHECK:STDOUT:   %.1: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %Op.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Add.%Op.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: %Op.type = fn_decl @Op [template = constants.%Op] {
// CHECK:STDOUT:     %Self.ref.loc5_15: %.1 = name_ref Self, %Self [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref.loc5_15 [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref.loc5_15, %.loc5_15.1 [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %self.loc5_9.1: @Op.%Self (%Self) = param self
// CHECK:STDOUT:     %self.loc5_9.2: @Op.%Self (%Self) = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:     %Self.ref.loc5_28: %.1 = name_ref Self, %Self [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_28.1: type = facet_type_access %Self.ref.loc5_28 [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_28.2: type = converted %Self.ref.loc5_28, %.loc5_28.1 [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %other.loc5_21.1: @Op.%Self (%Self) = param other
// CHECK:STDOUT:     %other.loc5_21.2: @Op.%Self (%Self) = bind_name other, %other.loc5_21.1
// CHECK:STDOUT:     %Self.ref.loc5_37: %.1 = name_ref Self, %Self [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_37.1: type = facet_type_access %Self.ref.loc5_37 [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_37.2: type = converted %Self.ref.loc5_37, %.loc5_37.1 [symbolic = @Op.%Self (constants.%Self)]
// CHECK:STDOUT:     %return.var: ref @Op.%Self (%Self) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_41: %.3 = assoc_entity element0, %Op.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_41
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(@Add.%Self: %.1) {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@Add.%self.loc5_9.2: @Op.%Self (%Self)](@Add.%other.loc5_21.2: @Op.%Self (%Self)) -> @Op.%Self (%Self);
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
// CHECK:STDOUT:   %.2: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %.2, %Op.type.2 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//a, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//a, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: %.4 = import_ref Main//a, inst+25, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.4: %Op.type.2 = import_ref Main//a, inst+19, loaded [template = constants.%Op.2]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//a, inst+19, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Add = imports.%import_ref.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %.loc6_7.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc10_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_10.2: type = converted %.loc10_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc10_6.1: %.1 = param x
// CHECK:STDOUT:     @F.%x: %.1 = bind_name x, %x.loc10_6.1
// CHECK:STDOUT:     %.loc10_17.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_17.2: type = converted %.loc10_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %y.loc10_13.1: %.1 = param y
// CHECK:STDOUT:     @F.%y: %.1 = bind_name y, %y.loc10_13.1
// CHECK:STDOUT:     %.loc10_24.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc10_24.2: type = converted %.loc10_24.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref %.1 = var <return slot>
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
// CHECK:STDOUT: impl @impl: %.1 as %.2 {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %Self.ref.loc7_15: type = name_ref Self, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %self.loc7_9.1: %.1 = param self
// CHECK:STDOUT:     %self.loc7_9.2: %.1 = bind_name self, %self.loc7_9.1
// CHECK:STDOUT:     %Self.ref.loc7_28: type = name_ref Self, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %other.loc7_21.1: %.1 = param other
// CHECK:STDOUT:     %other.loc7_21.2: %.1 = bind_name other, %other.loc7_21.1
// CHECK:STDOUT:     %Self.ref.loc7_37: type = name_ref Self, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc7_9.2: %.1](@impl.%other.loc7_21.2: %.1) -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_52: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_53: %.1 = converted %.loc7_52, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc7_53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %Self](%other: %Self) -> %Self;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x: %.1, %y: %.1) -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.1 = name_ref x, %x
// CHECK:STDOUT:   %Add.ref: type = name_ref Add, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   %Op.ref: %.4 = name_ref Op, imports.%import_ref.3 [template = constants.%.5]
// CHECK:STDOUT:   %.1: %Op.type.2 = interface_witness_access @impl.%.1, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc11_11: <bound method> = bound_method %x.ref, %.1
// CHECK:STDOUT:   %y.ref: %.1 = name_ref y, %y
// CHECK:STDOUT:   %Op.call: init %.1 = call %.loc11_11(%x.ref, %y.ref)
// CHECK:STDOUT:   %.loc11_20.1: ref %.1 = temporary_storage
// CHECK:STDOUT:   %.loc11_20.2: ref %.1 = temporary %.loc11_20.1, %Op.call
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_23: %.1 = converted %Op.call, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc11_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
