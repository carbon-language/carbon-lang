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
// CHECK:STDOUT:   %AddWith.type.bfd: type = facet_type <@AddWith, @AddWith(%C)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.45e: type = fn_type @AddWith.Op, @AddWith(%C) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness file.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.Op.type: type = fn_type @C.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.AddWith.impl.Op: %C.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.bfd = facet_value %C, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.133: type = facet_type <@AddAssignWith, @AddAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.Op.type.5af: type = fn_type @AddAssignWith.Op, @AddAssignWith(%C) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness: <witness> = impl_witness file.%AddAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.type.c84368.1: type = fn_type @C.as.AddAssignWith.impl.Op.loc27_34.1 [concrete]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.4b9a01.1: %C.as.AddAssignWith.impl.Op.type.c84368.1 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.133 = facet_value %C, (%AddAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.type.c84368.2: type = fn_type @C.as.AddAssignWith.impl.Op.loc27_34.2 [concrete]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.4b9a01.2: %C.as.AddAssignWith.impl.Op.type.c84368.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.7f4: type = fn_type_with_self_type %AddWith.Op.type.45e, %AddWith.facet [concrete]
// CHECK:STDOUT:   %.277: type = fn_type_with_self_type %AddAssignWith.Op.type.5af, %AddAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.7f4 = impl_witness_access constants.%AddWith.impl_witness, element1 [concrete = constants.%C.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.AddWith.impl.Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc30
// CHECK:STDOUT:   return %C.as.AddWith.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.277 = impl_witness_access constants.%AddAssignWith.impl_witness, element0 [concrete = constants.%C.as.AddAssignWith.impl.Op.4b9a01.2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %Op.ref: %C.as.AddAssignWith.impl.Op.type.c84368.1 = name_ref Op, @C.as.AddAssignWith.impl.%C.as.AddAssignWith.impl.Op.decl.loc27_34.1 [concrete = constants.%C.as.AddAssignWith.impl.Op.4b9a01.1]
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %.loc38, %Op.ref
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc38
// CHECK:STDOUT:   %C.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %C.as.AddAssignWith.impl.Op.bound(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
