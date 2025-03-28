// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/bit_xor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/bit_xor.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitXor {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.BitXorAssign {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  return a ^ b;
}

fn TestAssign(a: C*, b: C) {
  *a ^= b;
}

// CHECK:STDOUT: --- bit_xor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %BitXor.type: type = facet_type <@BitXor> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Op.type.e96: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %impl_witness.01d: <witness> = impl_witness (@impl.2e5.%Op.decl) [concrete]
// CHECK:STDOUT:   %Op.type.672: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.442: %Op.type.672 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXor.facet: %BitXor.type = facet_value %C, (%impl_witness.01d) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssign.type: type = facet_type <@BitXorAssign> [concrete]
// CHECK:STDOUT:   %Op.type.58d: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %impl_witness.8dc: <witness> = impl_witness (@impl.9ba.%Op.decl) [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.type.8ab: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.67d: %Op.type.8ab = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssign.facet: %BitXorAssign.type = facet_value %C, (%impl_witness.8dc) [concrete]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [concrete]
// CHECK:STDOUT:   %TestOp: %TestOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %.0d1: type = fn_type_with_self_type %Op.type.e96, %BitXor.facet [concrete]
// CHECK:STDOUT:   %TestAssign.type: type = fn_type @TestAssign [concrete]
// CHECK:STDOUT:   %TestAssign: %TestAssign.type = struct_value () [concrete]
// CHECK:STDOUT:   %.502: type = fn_type_with_self_type %Op.type.58d, %BitXorAssign.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitXor = %Core.BitXor
// CHECK:STDOUT:     .BitXorAssign = %Core.BitXorAssign
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitXor: type = import_ref Core//prelude/operators/bitwise, BitXor, loaded [concrete = constants.%BitXor.type]
// CHECK:STDOUT:   %Core.BitXorAssign: type = import_ref Core//prelude/operators/bitwise, BitXorAssign, loaded [concrete = constants.%BitXorAssign.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:     .TestAssign = %TestAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.2e5 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %BitXor.ref: type = name_ref BitXor, imports.%Core.BitXor [concrete = constants.%BitXor.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (@impl.2e5.%Op.decl) [concrete = constants.%impl_witness.01d]
// CHECK:STDOUT:   impl_decl @impl.9ba [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %BitXorAssign.ref: type = name_ref BitXorAssign, imports.%Core.BitXorAssign [concrete = constants.%BitXorAssign.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc22: <witness> = impl_witness (@impl.9ba.%Op.decl) [concrete = constants.%impl_witness.8dc]
// CHECK:STDOUT:   %TestOp.decl: %TestOp.type = fn_decl @TestOp [concrete = constants.%TestOp] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc26_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc26_14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc26_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign.decl: %TestAssign.type = fn_decl @TestAssign [concrete = constants.%TestAssign] {
// CHECK:STDOUT:     %a.patt: %ptr.019 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.019 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc30: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc30_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.019 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc30_25: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2e5: %C.ref as %BitXor.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.672 = fn_decl @Op.2 [concrete = constants.%Op.442] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc18_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc18_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc18_25: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9ba: %C.ref as %BitXorAssign.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.8ab = fn_decl @Op.4 [concrete = constants.%Op.67d] {
// CHECK:STDOUT:     %self.patt: %ptr.019 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.019 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc23_9: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %other.patt: %C = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %C = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc23_21: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc23_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc23_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %other: %C = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %C](%other.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc19_13.2: init %C = class_init (), %return [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc19_14: init %C = converted %.loc19_13.1, %.loc19_13.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc19_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr %self.param_patt: %ptr.019](%other.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param_patt: %C, %b.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.0d1 = impl_witness_access constants.%impl_witness.01d, element0 [concrete = constants.%Op.442]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %.loc26: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc26
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param_patt: %ptr.019, %b.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc31: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.502 = impl_witness_access constants.%impl_witness.8dc, element0 [concrete = constants.%Op.67d]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc31, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc31
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
