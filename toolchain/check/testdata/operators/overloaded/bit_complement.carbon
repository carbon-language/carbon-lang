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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/bit_complement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/bit_complement.carbon

// This file was generated from unary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitComplement where .Result = C {
  fn Op[self: C]() -> C {
    return {};
  }
}

fn TestOp(a: C) -> C {
  //@dump-sem-ir-begin
  return ^a;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- bit_complement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %BitComplement.type: type = facet_type <@BitComplement> [concrete]
// CHECK:STDOUT:   %Op.type.f25: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %BitComplement.impl_witness: <witness> = impl_witness file.%BitComplement.impl_witness_table [concrete]
// CHECK:STDOUT:   %Op.type.e57: type = fn_type @Op.loc22 [concrete]
// CHECK:STDOUT:   %Op.d4b: %Op.type.e57 = struct_value () [concrete]
// CHECK:STDOUT:   %BitComplement.facet.c4c: %BitComplement.type = facet_value %C, (%BitComplement.impl_witness) [concrete]
// CHECK:STDOUT:   %.18b: type = fn_type_with_self_type %Op.type.f25, %BitComplement.facet.c4c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1: %.18b = impl_witness_access constants.%BitComplement.impl_witness, element1 [concrete = constants.%Op.d4b]
// CHECK:STDOUT:   %bound_method.loc29: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.call: init %C = call %bound_method.loc29(%a.ref) to %.loc27
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
