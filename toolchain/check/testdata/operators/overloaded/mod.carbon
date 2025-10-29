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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/mod.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/mod.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.ModWith(C) where .Result = C {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.ModAssignWith(C) {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  //@dump-sem-ir-begin
  return a % b;
  //@dump-sem-ir-end
}

fn TestAssign(a: C*, b: C) {
  //@dump-sem-ir-begin
  *a %= b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- mod.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ModWith.type.a0d: type = facet_type <@ModWith, @ModWith(%C)> [concrete]
// CHECK:STDOUT:   %ModWith.Op.type.140: type = fn_type @ModWith.Op, @ModWith(%C) [concrete]
// CHECK:STDOUT:   %ModWith.impl_witness: <witness> = impl_witness file.%ModWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.ModWith.impl.Op.type: type = fn_type @C.as.ModWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.ModWith.impl.Op: %C.as.ModWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.facet: %ModWith.type.a0d = facet_value %C, (%ModWith.impl_witness) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.444: type = facet_type <@ModAssignWith, @ModAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.Op.type.dd0: type = fn_type @ModAssignWith.Op, @ModAssignWith(%C) [concrete]
// CHECK:STDOUT:   %ModAssignWith.impl_witness: <witness> = impl_witness file.%ModAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.ModAssignWith.impl.Op.type: type = fn_type @C.as.ModAssignWith.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.ModAssignWith.impl.Op: %C.as.ModAssignWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.444 = facet_value %C, (%ModAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.e00: type = fn_type_with_self_type %ModWith.Op.type.140, %ModWith.facet [concrete]
// CHECK:STDOUT:   %.a3f: type = fn_type_with_self_type %ModAssignWith.Op.type.dd0, %ModAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.e00 = impl_witness_access constants.%ModWith.impl_witness, element1 [concrete = constants.%C.as.ModWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.ModWith.impl.Op.call: init %C = call %bound_method(%a.ref, %b.ref) to %.loc30
// CHECK:STDOUT:   return %C.as.ModWith.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a.param: %ptr.019, %b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.019 = name_ref a, %a
// CHECK:STDOUT:   %.loc38: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem0: %.a3f = impl_witness_access constants.%ModAssignWith.impl_witness, element0 [concrete = constants.%C.as.ModAssignWith.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc38, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc38
// CHECK:STDOUT:   %C.as.ModAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
