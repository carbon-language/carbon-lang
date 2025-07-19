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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/add.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/add.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.AddWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.AddAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a + b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a += b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- add.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AddWith.type.ffb: type = facet_type <@AddWith, @AddWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.45e: type = fn_type @Op.1, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness file.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Op.type.638: type = fn_type @Op.loc22 [concrete]
// CHECK:STDOUT:   %Op.276: %Op.type.638 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet.3ef: %AddWith.type.ffb = facet_value %C, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.393: type = facet_type <@AddAssignWith, @AddAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.5af: type = fn_type @Op.6, @AddAssignWith(%C) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness: <witness> = impl_witness file.%AddAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.type.0d7: type = fn_type @Op.loc27 [concrete]
// CHECK:STDOUT:   %Op.572: %Op.type.0d7 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.393 = facet_value %C, (%AddAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.268: type = fn_type_with_self_type %Op.type.45e, %AddWith.facet.3ef [concrete]
// CHECK:STDOUT:   %.e2d: type = fn_type_with_self_type %Op.type.5af, %AddAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.268 = impl_witness_access constants.%AddWith.impl_witness, element1 [concrete = constants.%Op.276]
// CHECK:STDOUT:   %bound_method.loc32: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.call: init %C = call %bound_method.loc32(%a.ref, %b.ref) to %.loc30
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.e2d = impl_witness_access constants.%AddAssignWith.impl_witness, element0 [concrete = constants.%Op.572]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc38
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
