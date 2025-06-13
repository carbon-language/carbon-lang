// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
// CHECK:STDOUT:   %Other: type = bind_symbolic_name Other, 0 [symbolic]
// CHECK:STDOUT:   %ModWith.type.bcb: type = facet_type <@ModWith, @ModWith(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.9cc: %ModWith.type.bcb = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.type.4d0: type = facet_type <@ModWith, @ModWith(%C)> [concrete]
// CHECK:STDOUT:   %.Self.d23: %ModWith.type.4d0 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Op.type.140: type = fn_type @Op.1, @ModWith(%C) [concrete]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.d23 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.e42: <witness> = lookup_impl_witness %.Self.d23, @ModWith, @ModWith(%C) [symbolic_self]
// CHECK:STDOUT:   %ModWith.facet.d9a: %ModWith.type.4d0 = facet_value %.Self.as_type, (%ModWith.lookup_impl_witness.e42) [symbolic_self]
// CHECK:STDOUT:   %ModWith.impl_witness: <witness> = impl_witness file.%ModWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Op.type.0ad: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.d57: %Op.type.0ad = struct_value () [concrete]
// CHECK:STDOUT:   %ModWith.facet.517: %ModWith.type.4d0 = facet_value %C, (%ModWith.impl_witness) [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.8eb: type = facet_type <@ModAssignWith, @ModAssignWith(%C)> [concrete]
// CHECK:STDOUT:   %Op.type.dd0: type = fn_type @Op.7, @ModAssignWith(%C) [concrete]
// CHECK:STDOUT:   %ModAssignWith.impl_witness: <witness> = impl_witness file.%ModAssignWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.type.4dd: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.4f0: %Op.type.4dd = struct_value () [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet: %ModAssignWith.type.8eb = facet_value %C, (%ModAssignWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.f08: type = fn_type_with_self_type %Op.type.140, %ModWith.facet.517 [concrete]
// CHECK:STDOUT:   %.f3c: type = fn_type_with_self_type %Op.type.dd0, %ModAssignWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C, %b.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %impl.elem1: %.f08 = impl_witness_access constants.%ModWith.impl_witness, element1 [concrete = constants.%Op.d57]
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
// CHECK:STDOUT:   %impl.elem0: %.f3c = impl_witness_access constants.%ModAssignWith.impl_witness, element0 [concrete = constants.%Op.4f0]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc37, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc37
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%addr, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Result(constants.%Other, constants.%Self.9cc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Result(constants.%C, constants.%ModWith.facet.d9a) {}
// CHECK:STDOUT:
