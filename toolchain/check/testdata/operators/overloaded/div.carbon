// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/div.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/div.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.DivWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.DivAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a / b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a /= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DivWith.type.ead: type = facet_type <@DivWith, @DivWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.c4b: type = fn_type @Op.1, @DivWith(%C) [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness: <witness> = impl_witness file.%DivWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Op.type.a37: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.5b4: %Op.type.a37 = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.facet.f3b: %DivWith.type.ead = facet_value %C, (%DivWith.impl_witness) [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.08b: type = facet_type <@DivAssignWith, @DivAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.629: type = fn_type @Op.7, @DivAssignWith(%C) [concrete]
// CHECK:STDOUT:   %DivAssignWith.impl_witness: <witness> = impl_witness file.%DivAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.type.1b1: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.27f: %Op.type.1b1 = struct_value () [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet: %DivAssignWith.type.08b = facet_value %C, (%DivAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.bb6: type = fn_type_with_self_type %Op.type.c4b, %DivWith.facet.f3b [concrete]
// CHECK:STDOUT:   %.f19: type = fn_type_with_self_type %Op.type.629, %DivAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.bb6 = impl_witness_access constants.%DivWith.impl_witness, element1 [concrete = constants.%Op.5b4]
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
// CHECK:STDOUT:   %impl.elem0: %.f19 = impl_witness_access constants.%DivAssignWith.impl_witness, element0 [concrete = constants.%Op.27f]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc37, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc37
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
