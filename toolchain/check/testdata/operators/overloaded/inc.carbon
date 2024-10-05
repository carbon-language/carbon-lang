// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/inc.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/inc.carbon

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
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Inc.type: type = interface_type @Inc [template]
// CHECK:STDOUT:   %Self: %Inc.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %C [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %Self [symbolic]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [template]
// CHECK:STDOUT:   %TestOp: %TestOp.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %Inc.type, %Op.type.2 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Inc = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/arithmetic, inst+56, loaded [template = constants.%Inc.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/arithmetic, inst+58, unloaded
// CHECK:STDOUT:   %import_ref.3: %.8 = import_ref Core//prelude/operators/arithmetic, inst+75, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.4: %Op.type.2 = import_ref Core//prelude/operators/arithmetic, inst+69, loaded [template = constants.%Op.2]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/arithmetic, inst+69, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Inc.ref: type = name_ref Inc, imports.%import_ref.1 [template = constants.%Inc.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp.decl: %TestOp.type = fn_decl @TestOp [template = constants.%TestOp] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .Op = imports.%import_ref.3
// CHECK:STDOUT:   witness = (imports.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %Inc.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %self.patt: %.3 = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc18_21: type = ptr_type %C [template = constants.%.3]
// CHECK:STDOUT:     %self.param: %.3 = param self, runtime_param0
// CHECK:STDOUT:     %self: %.3 = bind_name self, %self.param
// CHECK:STDOUT:     %.loc18_9: %.3 = addr_pattern %self
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = interface_witness (%Op.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[addr %self: %.3]();
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self: %Inc.type) {
// CHECK:STDOUT:   %Self: %Inc.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.2.%Self (%Self) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: @Op.2.%.2 (%.5)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %.loc22_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc22_15.2: init %C = class_init (), %c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc22_16: init %C = converted %.loc22_15.1, %.loc22_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc22_16
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %Op.ref: %.8 = name_ref Op, imports.%import_ref.3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc23_3.1: %Op.type.2 = interface_witness_access constants.%.6, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc23_3.2: <bound method> = bound_method %c.ref, %.loc23_3.1
// CHECK:STDOUT:   %.loc23_5: %.3 = addr_of %c.ref
// CHECK:STDOUT:   %Op.call: init %.4 = call %.loc23_3.2(%.loc23_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%C) {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT:   %.2 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
