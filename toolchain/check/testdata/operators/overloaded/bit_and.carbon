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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/bit_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/bit_and.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitAndWith(C) where .Result = C {
  fn Op[unused self: C](unused other: C) -> C {
    return {};
  }
}
impl C as Core.BitAndAssignWith(C) {
  fn Op[unused ref self: C](unused other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a & b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a &= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- bit_and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.7a8: type = facet_type <@BitAndWith, @BitAndWith(%C)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness @C.as.BitAndWith.impl.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op.type: type = fn_type @C.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op: %C.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.7a8 = facet_value %C, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.c34: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%C, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.b04: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness: <witness> = impl_witness @C.as.BitAndAssignWith.impl.%BitAndAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.type: type = fn_type @C.as.BitAndAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op: %C.as.BitAndAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.b04 = facet_value %C, (%BitAndAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.21f: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%C, %BitAndAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.9d4: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.c34, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %ptr.ec1: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.3d4: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.21f, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.9d4 = impl_witness_access constants.%BitAndWith.impl_witness, element1 [concrete = constants.%C.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.BitAndWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.ec1, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.ec1 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.3d4 = impl_witness_access constants.%BitAndAssignWith.impl_witness, element0 [concrete = constants.%C.as.BitAndAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
