// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

// --- b.carbon

library "b" api;

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
// CHECK:STDOUT:   %Self: Add = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Add, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Add> = assoc_entity element0, @Add.%Op [template]
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
// CHECK:STDOUT:   %Self: Add = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op [template] {
// CHECK:STDOUT:     %Self.ref.loc5_15: Add = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref.loc5_15 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref.loc5_15, %.loc5_15.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc5_9.1: Self = param self
// CHECK:STDOUT:     %self.loc5_9.2: Self = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:     %Self.ref.loc5_28: Add = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_28.1: type = facet_type_access %Self.ref.loc5_28 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_28.2: type = converted %Self.ref.loc5_28, %.loc5_28.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %other.loc5_21.1: Self = param other
// CHECK:STDOUT:     %other.loc5_21.2: Self = bind_name other, %other.loc5_21.1
// CHECK:STDOUT:     %Self.ref.loc5_37: Add = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_37.1: type = facet_type_access %Self.ref.loc5_37 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_37.2: type = converted %Self.ref.loc5_37, %.loc5_37.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_41: <associated <function> in Add> = assoc_entity element0, %Op [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_41
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op[@Add.%self.loc5_9.2: Self](@Add.%other.loc5_21.2: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: Add = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (@impl.%Op) [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type @Add, <function> [template]
// CHECK:STDOUT:   %.5: <associated <function> in Add> = assoc_entity element0, file.%import_ref.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Add = %import_ref.1
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loc_13 [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Add> = import_ref ir1, inst+21, loc_58 [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+19, loc_14 [template = imports.%Op]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %.loc6_7.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %Add.decl: type = interface_decl @Add [template = constants.%.2] {}
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %.loc10_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc10_10.2: type = converted %.loc10_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc10_6.1: () = param x
// CHECK:STDOUT:     @F.%x: () = bind_name x, %x.loc10_6.1
// CHECK:STDOUT:     %.loc10_17.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc10_17.2: type = converted %.loc10_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %y.loc10_13.1: () = param y
// CHECK:STDOUT:     @F.%y: () = bind_name y, %y.loc10_13.1
// CHECK:STDOUT:     %.loc10_24.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc10_24.2: type = converted %.loc10_24.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: () as Add {
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %Self.ref.loc7_15: type = name_ref Self, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %self.loc7_9.1: () = param self
// CHECK:STDOUT:     %self.loc7_9.2: () = bind_name self, %self.loc7_9.1
// CHECK:STDOUT:     %Self.ref.loc7_28: type = name_ref Self, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %other.loc7_21.1: () = param other
// CHECK:STDOUT:     %other.loc7_21.2: () = bind_name other, %other.loc7_21.1
// CHECK:STDOUT:     %Self.ref.loc7_37: type = name_ref Self, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc7_9.2: ()](@impl.%other.loc7_21.2: ()) -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_52: () = tuple_literal ()
// CHECK:STDOUT:   %tuple: () = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_53: () = converted %.loc7_52, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc7_53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self](%other: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x: (), %y: ()) -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: () = name_ref x, %x
// CHECK:STDOUT:   %Add.ref: type = name_ref Add, file.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   %Op.ref: <associated <function> in Add> = name_ref Op, file.%import_ref.2 [template = constants.%.5]
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%Op]
// CHECK:STDOUT:   %.loc11_11: <bound method> = bound_method %x.ref, %.1
// CHECK:STDOUT:   %y.ref: () = name_ref y, %y
// CHECK:STDOUT:   %.loc11_20.1: init () = call %.loc11_11(%x.ref, %y.ref)
// CHECK:STDOUT:   %.loc11_20.2: ref () = temporary_storage
// CHECK:STDOUT:   %.loc11_20.3: ref () = temporary %.loc11_20.2, %.loc11_20.1
// CHECK:STDOUT:   %tuple: () = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_23: () = converted %.loc11_20.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc11_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
