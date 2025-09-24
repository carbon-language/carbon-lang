// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
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
// CHECK:STDOUT:   %BitXorWith.type.bf8: type = facet_type <@BitXorWith, @BitXorWith(%C)> [concrete]
// CHECK:STDOUT:   %BitXorWith.Op.type.502: type = fn_type @BitXorWith.Op, @BitXorWith(%C) [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness: <witness> = impl_witness file.%BitXorWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op.type: type = fn_type @C.as.BitXorWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op: %C.as.BitXorWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.facet: %BitXorWith.type.bf8 = facet_value %C, (%BitXorWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.38a: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.Op.type.ad0: type = fn_type @BitXorAssignWith.Op, @BitXorAssignWith(%C) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness: <witness> = impl_witness file.%BitXorAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op.type: type = fn_type @C.as.BitXorAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op: %C.as.BitXorAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.38a = facet_value %C, (%BitXorAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.a7b: type = fn_type_with_self_type %BitXorWith.Op.type.502, %BitXorWith.facet [concrete]
// CHECK:STDOUT:   %.d84: type = fn_type_with_self_type %BitXorAssignWith.Op.type.ad0, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.a7b = impl_witness_access constants.%BitXorWith.impl_witness, element1 [concrete = constants.%C.as.BitXorWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc30
// CHECK:STDOUT:   return %C.as.BitXorWith.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.d84 = impl_witness_access constants.%BitXorAssignWith.impl_witness, element0 [concrete = constants.%C.as.BitXorAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc38
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
