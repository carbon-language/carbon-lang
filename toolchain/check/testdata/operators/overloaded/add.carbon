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
  fn Op[unused self: C](unused other: C) -> C {
    return {};
  }
}
impl C as Core.AddAssignWith(C) {
  fn Op[unused ref self: C](unused other: C) {}
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
// CHECK:STDOUT:   %AddWith.type.2bd: type = facet_type <@AddWith, @AddWith(%C)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness @C.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.Op.type: type = fn_type @C.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.Op: %C.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.2bd = facet_value %C, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.c2b: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%C, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.cba: type = facet_type <@AddAssignWith, @AddAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness: <witness> = impl_witness @C.as.AddAssignWith.impl.%AddAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.type: type = fn_type @C.as.AddAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op: %C.as.AddAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.cba = facet_value %C, (%AddAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.62f: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%C, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.757: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.c2b, %AddWith.facet [concrete]
// CHECK:STDOUT:   %ptr.00e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.fbc: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.62f, %AddAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.757 = impl_witness_access constants.%AddWith.impl_witness, element1 [concrete = constants.%C.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.AddWith.impl.Op.call: init %C to %.loc30_26.1 = call %bound_method(%a.ref, %b.ref)
// CHECK:STDOUT:   return %C.as.AddWith.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.00e, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.00e = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.fbc = impl_witness_access constants.%AddAssignWith.impl_witness, element0 [concrete = constants.%C.as.AddAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc38, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
