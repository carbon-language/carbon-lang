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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/negate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/negate.carbon

// This file was generated from unary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Negate where .Result = C {
  fn Op[unused self: C]() -> C {
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
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness @C.as.Negate.impl.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.Negate.impl.Op.type: type = fn_type @C.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Negate.impl.Op: %C.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %C, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.0da: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.962: type = fn_type_with_self_type %Negate.WithSelf.Op.type.0da, %Negate.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1: %.962 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%C.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.Negate.impl.Op.call: init %C to %.loc27_20.1 = call %bound_method(%a.ref)
// CHECK:STDOUT:   return %C.as.Negate.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
