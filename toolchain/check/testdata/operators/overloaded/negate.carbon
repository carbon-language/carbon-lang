// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/negate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/negate.carbon

// This file was generated from unary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Negate {
  fn Op[self: C]() -> C {
    return {};
  }
}

fn TestOp(a: C) -> C {
  return -a;
}

// CHECK:STDOUT: --- negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [template]
// CHECK:STDOUT:   %TestOp: %TestOp.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Negate, %Op.type.2 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Negate = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/arithmetic, inst+72, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/arithmetic, inst+74, unloaded
// CHECK:STDOUT:   %import_ref.3: %.6 = import_ref Core//prelude/operators/arithmetic, inst+90, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.4: %Op.type.2 = import_ref Core//prelude/operators/arithmetic, inst+85, loaded [template = constants.%Op.2]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/arithmetic, inst+85, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Negate.ref: type = name_ref Negate, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp.decl: %TestOp.type = fn_decl @TestOp [template = constants.%TestOp] {
// CHECK:STDOUT:     %C.ref.loc23_14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc23_11.1: %C = param a
// CHECK:STDOUT:     @TestOp.%a: %C = bind_name a, %a.loc23_11.1
// CHECK:STDOUT:     %C.ref.loc23_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestOp.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .Op = imports.%import_ref.3
// CHECK:STDOUT:   witness = (imports.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.2 {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %C.ref.loc18_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc18_9.1: %C = param self
// CHECK:STDOUT:     %self.loc18_9.2: %C = bind_name self, %self.loc18_9.1
// CHECK:STDOUT:     %C.ref.loc18_23: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.4]
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
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc18_9.2: %C]() -> @impl.%return.var: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc19_13.2: init %C = class_init (), @impl.%return.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_14: init %C = converted %.loc19_13.1, %.loc19_13.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc19_14 to @impl.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %Self]() -> %Self;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %.1: %Op.type.2 = interface_witness_access @impl.%.1, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc24: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc23: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Op.call: init %C = call %.loc24(%a.ref) to %.loc23
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
