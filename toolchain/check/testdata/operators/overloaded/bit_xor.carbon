// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/bit_xor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/bit_xor.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitXorWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.BitXorAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a ^ b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a ^= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- bit_xor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitXorWith.type.f9b: type = facet_type <@BitXorWith, @BitXorWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.502: type = fn_type @Op.1, @BitXorWith(%C) [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness: <witness> = impl_witness file.%BitXorWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Op.type.5e4: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.ef1: %Op.type.5e4 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.d57: %BitXorWith.type.f9b = facet_value %C, (%BitXorWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.57a: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.ad0: type = fn_type @Op.7, @BitXorAssignWith(%C) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness: <witness> = impl_witness file.%BitXorAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.type.d64: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.5e5: %Op.type.d64 = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.57a = facet_value %C, (%BitXorAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.c60: type = fn_type_with_self_type %Op.type.502, %BitXorWith.facet.d57 [concrete]
// CHECK:STDOUT:   %.d8a: type = fn_type_with_self_type %Op.type.ad0, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.c60 = impl_witness_access constants.%BitXorWith.impl_witness, element1 [concrete = constants.%Op.ef1]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc29
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc37: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.d8a = impl_witness_access constants.%BitXorAssignWith.impl_witness, element0 [concrete = constants.%Op.5e5]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc37, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc37
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
