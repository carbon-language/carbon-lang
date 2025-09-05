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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/bit_or.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/bit_or.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitOrWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.BitOrAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a | b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a |= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- bit_or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitOrWith.type.502: type = facet_type <@BitOrWith, @BitOrWith(%C)> [concrete]
// CHECK:STDOUT:   %BitOrWith.Op.type.501: type = fn_type @BitOrWith.Op, @BitOrWith(%C) [concrete]
// CHECK:STDOUT:   %BitOrWith.impl_witness: <witness> = impl_witness file.%BitOrWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitOrWith.impl.Op.type: type = fn_type @C.as.BitOrWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitOrWith.impl.Op: %C.as.BitOrWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.f19: %BitOrWith.type.502 = facet_value %C, (%BitOrWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.3e6: type = facet_type <@BitOrAssignWith, @BitOrAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.Op.type.efc: type = fn_type @BitOrAssignWith.Op, @BitOrAssignWith(%C) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.impl_witness: <witness> = impl_witness file.%BitOrAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitOrAssignWith.impl.Op.type: type = fn_type @C.as.BitOrAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitOrAssignWith.impl.Op: %C.as.BitOrAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet: %BitOrAssignWith.type.3e6 = facet_value %C, (%BitOrAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.b8d: type = fn_type_with_self_type %BitOrWith.Op.type.501, %BitOrWith.facet.f19 [concrete]
// CHECK:STDOUT:   %.c8a: type = fn_type_with_self_type %BitOrAssignWith.Op.type.efc, %BitOrAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.b8d = impl_witness_access constants.%BitOrWith.impl_witness, element1 [concrete = constants.%C.as.BitOrWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitOrWith.impl.Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc30
// CHECK:STDOUT:   return %C.as.BitOrWith.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.c8a = impl_witness_access constants.%BitOrAssignWith.impl_witness, element0 [concrete = constants.%C.as.BitOrAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc38
// CHECK:STDOUT:   %C.as.BitOrAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
