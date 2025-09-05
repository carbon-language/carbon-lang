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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/inc.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/inc.carbon

// This file was generated from unary_stmt.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Inc {
  fn Op[addr self: C*]();
}

fn TestOp() {
  var c: C = {};
  //@dump-sem-ir-begin
  ++c;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- inc.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %Inc.Op.type: type = fn_type @Inc.Op [concrete]
// CHECK:STDOUT:   %Inc.impl_witness: <witness> = impl_witness file.%Inc.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.Inc.impl.Op.type: type = fn_type @C.as.Inc.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Inc.impl.Op: %C.as.Inc.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inc.facet: %Inc.type = facet_value %C, (%Inc.impl_witness) [concrete]
// CHECK:STDOUT:   %.bc7: type = fn_type_with_self_type %Inc.Op.type, %Inc.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %impl.elem0: %.bc7 = impl_witness_access constants.%Inc.impl_witness, element0 [concrete = constants.%C.as.Inc.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %addr.loc28: %ptr.019 = addr_of %c.ref
// CHECK:STDOUT:   %C.as.Inc.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc28)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
