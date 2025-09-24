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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/right_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/right_shift.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.RightShiftWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.RightShiftAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a >> b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a >>= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- right_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %RightShiftWith.type.a33: type = facet_type <@RightShiftWith, @RightShiftWith(%C)> [concrete]
// CHECK:STDOUT:   %RightShiftWith.Op.type.19b: type = fn_type @RightShiftWith.Op, @RightShiftWith(%C) [concrete]
// CHECK:STDOUT:   %RightShiftWith.impl_witness: <witness> = impl_witness file.%RightShiftWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.RightShiftWith.impl.Op.type: type = fn_type @C.as.RightShiftWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.RightShiftWith.impl.Op: %C.as.RightShiftWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet: %RightShiftWith.type.a33 = facet_value %C, (%RightShiftWith.impl_witness) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.type.df5: type = facet_type <@RightShiftAssignWith, @RightShiftAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.Op.type.ae9: type = fn_type @RightShiftAssignWith.Op, @RightShiftAssignWith(%C) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.impl_witness: <witness> = impl_witness file.%RightShiftAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.RightShiftAssignWith.impl.Op.type: type = fn_type @C.as.RightShiftAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.RightShiftAssignWith.impl.Op: %C.as.RightShiftAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet: %RightShiftAssignWith.type.df5 = facet_value %C, (%RightShiftAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.bc2: type = fn_type_with_self_type %RightShiftWith.Op.type.19b, %RightShiftWith.facet [concrete]
// CHECK:STDOUT:   %.a3c: type = fn_type_with_self_type %RightShiftAssignWith.Op.type.ae9, %RightShiftAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.bc2 = impl_witness_access constants.%RightShiftWith.impl_witness, element1 [concrete = constants.%C.as.RightShiftWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.RightShiftWith.impl.Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc30
// CHECK:STDOUT:   return %C.as.RightShiftWith.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.a3c = impl_witness_access constants.%RightShiftAssignWith.impl_witness, element0 [concrete = constants.%C.as.RightShiftAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc38
// CHECK:STDOUT:   %C.as.RightShiftAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
