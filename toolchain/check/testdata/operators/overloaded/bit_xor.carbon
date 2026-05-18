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
  fn Op[unused self: C](unused other: C) -> C {
    return {};
  }
}
impl C as Core.BitXorAssignWith(C) {
  fn Op[unused ref self: C](unused other: C) {}
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
// CHECK:STDOUT:   %BitXorWith.type.732: type = facet_type <@BitXorWith, @BitXorWith(%C)> [concrete]
// CHECK:STDOUT:   %BitXorWith.impl_witness: <witness> = impl_witness @C.as.BitXorWith.impl.%BitXorWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op.type: type = fn_type @C.as.BitXorWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op: %C.as.BitXorWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorWith.facet: %BitXorWith.type.732 = facet_value %C, (%BitXorWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.34f: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%C, %BitXorWith.facet) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.type.697: type = facet_type <@BitXorAssignWith, @BitXorAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.impl_witness: <witness> = impl_witness @C.as.BitXorAssignWith.impl.%BitXorAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op.type: type = fn_type @C.as.BitXorAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op: %C.as.BitXorAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet: %BitXorAssignWith.type.697 = facet_value %C, (%BitXorAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.309: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%C, %BitXorAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.6c1: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.34f, %BitXorWith.facet [concrete]
// CHECK:STDOUT:   %ptr.ec1: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.c76: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.309, %BitXorAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.6c1 = impl_witness_access constants.%BitXorWith.impl_witness, element1 [concrete = constants.%C.as.BitXorWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitXorWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.BitXorWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.ec1, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.ec1 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.c76 = impl_witness_access constants.%BitXorAssignWith.impl_witness, element0 [concrete = constants.%C.as.BitXorAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.BitXorAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
