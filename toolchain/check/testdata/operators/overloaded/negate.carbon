// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/negate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/negate.carbon

// This file was generated from unary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Negate where .Result = C {
  fn Op[self: C]() -> C {
    return {};
  }
}

fn TestOp(a: C) -> C {
  //@dump-sem-ir-begin
  return -a;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness file.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Op.type.def: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.045: %Op.type.def = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet.b50: %Negate.type = facet_value %C, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %.26d: type = fn_type_with_self_type %Op.type.e42, %Negate.facet.b50 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1: %.26d = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Op.045]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.call: init %C = call %bound_method(%a.ref) to %.loc26
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
