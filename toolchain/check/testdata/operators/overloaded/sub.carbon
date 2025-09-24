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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/sub.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/sub.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.SubWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.SubAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a - b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a -= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %SubWith.type.c5d: type = facet_type <@SubWith, @SubWith(%C)> [concrete]
// CHECK:STDOUT:   %SubWith.Op.type.fc0: type = fn_type @SubWith.Op, @SubWith(%C) [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness: <witness> = impl_witness file.%SubWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.SubWith.impl.Op.type: type = fn_type @C.as.SubWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.SubWith.impl.Op: %C.as.SubWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.facet: %SubWith.type.c5d = facet_value %C, (%SubWith.impl_witness) [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.9f4: type = facet_type <@SubAssignWith, @SubAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.Op.type.ffe: type = fn_type @SubAssignWith.Op, @SubAssignWith(%C) [concrete]
// CHECK:STDOUT:   %SubAssignWith.impl_witness: <witness> = impl_witness file.%SubAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.SubAssignWith.impl.Op.type: type = fn_type @C.as.SubAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.SubAssignWith.impl.Op: %C.as.SubAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet: %SubAssignWith.type.9f4 = facet_value %C, (%SubAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.c27: type = fn_type_with_self_type %SubWith.Op.type.fc0, %SubWith.facet [concrete]
// CHECK:STDOUT:   %.52c: type = fn_type_with_self_type %SubAssignWith.Op.type.ffe, %SubAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.c27 = impl_witness_access constants.%SubWith.impl_witness, element1 [concrete = constants.%C.as.SubWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.SubWith.impl.Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc30
// CHECK:STDOUT:   return %C.as.SubWith.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.52c = impl_witness_access constants.%SubAssignWith.impl_witness, element0 [concrete = constants.%C.as.SubAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc38
// CHECK:STDOUT:   %C.as.SubAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
