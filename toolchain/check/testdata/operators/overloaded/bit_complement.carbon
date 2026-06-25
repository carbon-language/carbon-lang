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
  fn Op(unused self: C) -> C {
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
// CHECK:STDOUT:   %BitComplement.impl_witness: <witness> = impl_witness @C.as.BitComplement.impl.%BitComplement.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.BitComplement.impl.Op.type: type = fn_type @C.as.BitComplement.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.BitComplement.impl.Op: %C.as.BitComplement.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitComplement.facet: %BitComplement.type = facet_value %C, (%BitComplement.impl_witness) [concrete]
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.type.6db: type = fn_type @BitComplement.WithSelf.Op, @BitComplement.WithSelf(%BitComplement.facet) [concrete]
// CHECK:STDOUT:   %.827: type = fn_type_with_self_type %BitComplement.WithSelf.Op.type.6db, %BitComplement.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a.param: %C) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %impl.elem1: %.827 = impl_witness_access constants.%BitComplement.impl_witness, element1 [concrete = constants.%C.as.BitComplement.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.BitComplement.impl.Op.call: init %C to %.loc27_20.1 = call %bound_method(%a.ref)
// CHECK:STDOUT:   return %C.as.BitComplement.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
