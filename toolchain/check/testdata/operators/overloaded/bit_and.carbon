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
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.BitAndAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
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
// CHECK:STDOUT:   %BitAndWith.type.488: type = facet_type <@BitAndWith, @BitAndWith(%C)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.914: type = fn_type @BitAndWith.Op, @BitAndWith(%C) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness file.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op.type: type = fn_type @C.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op: %C.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.488 = facet_value %C, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.bc6: type = facet_type <@BitAndAssignWith, @BitAndAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.Op.type.73d: type = fn_type @BitAndAssignWith.Op, @BitAndAssignWith(%C) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.impl_witness: <witness> = impl_witness file.%BitAndAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.type.d236d7.1: type = fn_type @C.as.BitAndAssignWith.impl.Op.loc27_34.1 [concrete]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.83e29f.1: %C.as.BitAndAssignWith.impl.Op.type.d236d7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet: %BitAndAssignWith.type.bc6 = facet_value %C, (%BitAndAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.type.d236d7.2: type = fn_type @C.as.BitAndAssignWith.impl.Op.loc27_34.2 [concrete]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.83e29f.2: %C.as.BitAndAssignWith.impl.Op.type.d236d7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.f52: type = fn_type_with_self_type %BitAndWith.Op.type.914, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %.ef9: type = fn_type_with_self_type %BitAndAssignWith.Op.type.73d, %BitAndAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.f52 = impl_witness_access constants.%BitAndWith.impl_witness, element1 [concrete = constants.%C.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitAndWith.impl.Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc30
// CHECK:STDOUT:   return %C.as.BitAndWith.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.ef9 = impl_witness_access constants.%BitAndAssignWith.impl_witness, element0 [concrete = constants.%C.as.BitAndAssignWith.impl.Op.83e29f.2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %Op.ref: %C.as.BitAndAssignWith.impl.Op.type.d236d7.1 = name_ref Op, @C.as.BitAndAssignWith.impl.%C.as.BitAndAssignWith.impl.Op.decl.loc27_34.1 [concrete = constants.%C.as.BitAndAssignWith.impl.Op.83e29f.1]
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.bound: <bound method> = bound_method %.loc38, %Op.ref
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc38
// CHECK:STDOUT:   %C.as.BitAndAssignWith.impl.Op.call: init %empty_tuple.type = call %C.as.BitAndAssignWith.impl.Op.bound(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
