// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/left_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/left_shift.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.LeftShift {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.LeftShiftAssign {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  return a << b;
}

fn TestAssign(a: C*, b: C) {
  *a <<= b;
}

// CHECK:STDOUT: --- left_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %LeftShift.type: type = facet_type <@LeftShift> [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %LeftShiftAssign.type: type = facet_type <@LeftShiftAssign> [template]
// CHECK:STDOUT:   %.7: type = ptr_type %C [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %.11: <witness> = interface_witness (%Op.3) [template]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [template]
// CHECK:STDOUT:   %TestOp: %TestOp.type = struct_value () [template]
// CHECK:STDOUT:   %TestAssign.type: type = fn_type @TestAssign [template]
// CHECK:STDOUT:   %TestAssign: %TestAssign.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .LeftShift = %import_ref.1
// CHECK:STDOUT:     .LeftShiftAssign = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/bitwise, inst+224, loaded [template = constants.%LeftShift.type]
// CHECK:STDOUT:   %import_ref.5: type = import_ref Core//prelude/operators/bitwise, inst+258, loaded [template = constants.%LeftShiftAssign.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:     .TestAssign = %TestAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %LeftShift.ref: type = name_ref LeftShift, imports.%import_ref.1 [template = constants.%LeftShift.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %LeftShiftAssign.ref: type = name_ref LeftShiftAssign, imports.%import_ref.5 [template = constants.%LeftShiftAssign.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp.decl: %TestOp.type = fn_decl @TestOp [template = constants.%TestOp] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc26_14: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc26_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc26_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign.decl: %TestAssign.type = fn_decl @TestAssign [template = constants.%TestAssign] {
// CHECK:STDOUT:     %a.patt: %.7 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.7 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc30_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc30: type = ptr_type %C [template = constants.%.7]
// CHECK:STDOUT:     %C.ref.loc30_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.7 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C.ref as %LeftShift.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc18_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc18_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc18_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = interface_witness (%Op.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C.ref as %LeftShiftAssign.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.3 = fn_decl @Op.3 [template = constants.%Op.3] {
// CHECK:STDOUT:     %self.patt: %.7 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.7 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc23_9: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc23_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc23_21: type = ptr_type %C [template = constants.%.7]
// CHECK:STDOUT:     %C.ref.loc23_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.param: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.7 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: <witness> = interface_witness (%Op.decl) [template = constants.%.11]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[%self.param_patt: %C](%other.param_patt: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc19_13.2: init %C = class_init (), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_14: init %C = converted %.loc19_13.1, %.loc19_13.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc19_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.3[addr %self.param_patt: %.7](%other.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param_patt: %C, %b.param_patt: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.loc27_12.1: %Op.type.2 = interface_witness_access constants.%.5, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc27_12.2: <bound method> = bound_method %a.ref, %.loc27_12.1
// CHECK:STDOUT:   %.loc26_23.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Op.call: init %C = call %.loc27_12.2(%a.ref, %b.ref) to %.loc26_23.2
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param_patt: %.7, %b.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.7 = name_ref a, %a
// CHECK:STDOUT:   %.loc31_3.1: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.loc31_6.1: %Op.type.4 = interface_witness_access constants.%.11, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %.loc31_6.2: <bound method> = bound_method %.loc31_3.1, %.loc31_6.1
// CHECK:STDOUT:   %.loc31_3.2: %.7 = addr_of %.loc31_3.1
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %.loc31_6.2(%.loc31_3.2, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
