// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/left_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/left_shift.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.LeftShiftWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.LeftShiftAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a << b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a <<= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- left_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.type.f92: type = facet_type <@LeftShiftWith, @LeftShiftWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.097: type = fn_type @Op.1, @LeftShiftWith(%C) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.impl_witness: <witness> = impl_witness file.%LeftShiftWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Op.type.be6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.4d9: %Op.type.be6 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.662: %LeftShiftWith.type.f92 = facet_value %C, (%LeftShiftWith.impl_witness) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.56d: type = facet_type <@LeftShiftAssignWith, @LeftShiftAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.30f: type = fn_type @Op.7, @LeftShiftAssignWith(%C) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.impl_witness: <witness> = impl_witness file.%LeftShiftAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.type.b1b6: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.07c: %Op.type.b1b6 = struct_value () [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet: %LeftShiftAssignWith.type.56d = facet_value %C, (%LeftShiftAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.cd9: type = fn_type_with_self_type %Op.type.097, %LeftShiftWith.facet.662 [concrete]
// CHECK:STDOUT:   %.97b: type = fn_type_with_self_type %Op.type.30f, %LeftShiftAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.cd9 = impl_witness_access constants.%LeftShiftWith.impl_witness, element1 [concrete = constants.%Op.4d9]
// CHECK:STDOUT:   %bound_method.loc31: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.call: init %C = call %bound_method.loc31(%a.ref, %b.ref) to %.loc29
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc37: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.97b = impl_witness_access constants.%LeftShiftAssignWith.impl_witness, element0 [concrete = constants.%Op.07c]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc37, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc37
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
