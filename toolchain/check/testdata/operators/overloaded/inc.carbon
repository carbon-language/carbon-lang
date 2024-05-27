// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// This file was generated from unary_stmt.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Inc {
  fn Op[addr self: C*]();
}

fn TestOp() {
  var c: C = {};
  ++c;
}

// CHECK:STDOUT: --- inc.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Inc [template]
// CHECK:STDOUT:   %Self: Inc = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type C [template]
// CHECK:STDOUT:   %Op.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Op = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %Op.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %struct.2: Op = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%struct.1) [template]
// CHECK:STDOUT:   %TestOp: type = fn_type @TestOp [template]
// CHECK:STDOUT:   %struct.3: TestOp = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct.4: C = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type @Inc, Op [template]
// CHECK:STDOUT:   %.9: <associated Op in Inc> = assoc_entity element0, file.%import_ref.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir4, inst+49, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir4, inst+51, unloaded
// CHECK:STDOUT:   %import_ref.3: <associated Op in Inc> = import_ref ir4, inst+65, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.4: Op = import_ref ir4, inst+61, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Inc.ref: type = name_ref Inc, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp.decl: TestOp = fn_decl @TestOp [template = constants.%struct.3] {}
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir4, inst+49, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir4, inst+61, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .Op = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Inc {
// CHECK:STDOUT:   %Op.decl: Op = fn_decl @Op.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc14_21: type = ptr_type C [template = constants.%.3]
// CHECK:STDOUT:     %self.loc14_14.1: C* = param self
// CHECK:STDOUT:     %self.loc14_14.3: C* = bind_name self, %self.loc14_14.1
// CHECK:STDOUT:     %.loc14_9: C* = addr_pattern %self.loc14_14.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[addr @impl.%self.loc14_14.3: C*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[addr %self: Self*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %.loc18_15.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc18_15.2: init C = class_init (), %c.var [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc18_16: init C = converted %.loc18_15.1, %.loc18_15.2 [template = constants.%struct.4]
// CHECK:STDOUT:   assign %c.var, %.loc18_16
// CHECK:STDOUT:   %c.ref: ref C = name_ref c, %c
// CHECK:STDOUT:   %.1: Op = interface_witness_access @impl.%.1, element0 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc19_3: <bound method> = bound_method %c.ref, %.1
// CHECK:STDOUT:   %.loc19_5: C* = addr_of %c.ref
// CHECK:STDOUT:   %Op.call: init () = call %.loc19_3(%.loc19_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
