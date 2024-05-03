// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- core.carbon

package Core api;

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

// --- user.carbon

import Core;

// TODO: This should be in `Core`, but currently impl lookup only looks in the
// current file.
impl i32 as Core.Add {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

var arr: [i32; 1 + 2] = (3, 4, 3 + 4);

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: Add = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op: type = fn_type @Op [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Op = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Add, Op [template]
// CHECK:STDOUT:   %.4: <associated Op in Add> = assoc_entity element0, @Add.%Op.decl [template]
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
// CHECK:STDOUT:   %Op.decl: Op = fn_decl @Op [template = constants.%struct] {
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
// CHECK:STDOUT:   %.loc5_41: <associated Op in Add> = assoc_entity element0, %Op.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_41
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op[@Add.%self.loc5_9.2: Self](@Add.%other.loc5_21.2: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self: Add = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Op = struct_value () [template]
// CHECK:STDOUT:   %Op.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %struct.2: Op = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (%struct.1) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Add, Op [template]
// CHECK:STDOUT:   %.7: <associated Op in Add> = assoc_entity element0, file.%import_ref.6 [template]
// CHECK:STDOUT:   %.8: <bound method> = bound_method %.4, %struct.1 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.10: type = array_type %.9, i32 [template]
// CHECK:STDOUT:   %.11: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.13: <bound method> = bound_method %.9, %struct.1 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.15: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: [i32; 3] = tuple_value (%.9, %.12, %.14) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.2: <associated Op in Add> = import_ref ir2, inst+24, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.4: Op = import_ref ir2, inst+19, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, %import_ref.1 [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_20: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir2, inst+1, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir2, inst+19, unloaded
// CHECK:STDOUT:   %.1: Op = interface_witness_access @impl.%.1, element0 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_18: <bound method> = bound_method %.loc10_16, %.1 [template = constants.%.8]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc10_18(%.loc10_16, %.loc10_20) [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_21: type = array_type %int.sadd, i32 [template = constants.%.10]
// CHECK:STDOUT:   %arr.var: ref [i32; 3] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 3] = bind_name arr, %arr.var
// CHECK:STDOUT:   %import_ref.7: type = import_ref ir2, inst+1, loaded [template = constants.%.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as Add {
// CHECK:STDOUT:   %Op.decl: Op = fn_decl @Op.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref.loc7_15: type = name_ref Self, i32 [template = i32]
// CHECK:STDOUT:     %self.loc7_9.1: i32 = param self
// CHECK:STDOUT:     %self.loc7_9.2: i32 = bind_name self, %self.loc7_9.1
// CHECK:STDOUT:     %Self.ref.loc7_28: type = name_ref Self, i32 [template = i32]
// CHECK:STDOUT:     %other.loc7_21.1: i32 = param other
// CHECK:STDOUT:     %other.loc7_21.2: i32 = bind_name other, %other.loc7_21.1
// CHECK:STDOUT:     %Self.ref.loc7_37: type = name_ref Self, i32 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc7_9.2: i32](@impl.%other.loc7_21.2: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self](%other: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_26: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_29: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:   %.loc10_32: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_36: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:   %.1: Op = interface_witness_access @impl.%.1, element0 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_34: <bound method> = bound_method %.loc10_32, %.1 [template = constants.%.13]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc10_34(%.loc10_32, %.loc10_36) [template = constants.%.14]
// CHECK:STDOUT:   %.loc10_37.1: (i32, i32, i32) = tuple_literal (%.loc10_26, %.loc10_29, %int.sadd)
// CHECK:STDOUT:   %.loc10_37.2: i32 = int_literal 0 [template = constants.%.16]
// CHECK:STDOUT:   %.loc10_37.3: ref i32 = array_index file.%arr.var, %.loc10_37.2
// CHECK:STDOUT:   %.loc10_37.4: init i32 = initialize_from %.loc10_26 to %.loc10_37.3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_37.5: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_37.6: ref i32 = array_index file.%arr.var, %.loc10_37.5
// CHECK:STDOUT:   %.loc10_37.7: init i32 = initialize_from %.loc10_29 to %.loc10_37.6 [template = constants.%.12]
// CHECK:STDOUT:   %.loc10_37.8: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_37.9: ref i32 = array_index file.%arr.var, %.loc10_37.8
// CHECK:STDOUT:   %.loc10_37.10: init i32 = initialize_from %int.sadd to %.loc10_37.9 [template = constants.%.14]
// CHECK:STDOUT:   %.loc10_37.11: init [i32; 3] = array_init (%.loc10_37.4, %.loc10_37.7, %.loc10_37.10) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc10_38: init [i32; 3] = converted %.loc10_37.1, %.loc10_37.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc10_38
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
