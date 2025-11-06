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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/dec.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/dec.carbon

// This file was generated from unary_stmt.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Dec {
  fn Op[addr self: C*]();
}

fn TestOp() {
  var c: C = {};
  //@dump-sem-ir-begin
  --c;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- dec.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.Op.type: type = fn_type @Dec.Op [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness file.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.Dec.impl.Op.type.987e49.1: type = fn_type @C.as.Dec.impl.Op.loc22_25.1 [concrete]
// CHECK:STDOUT:   %C.as.Dec.impl.Op.26619b.1: %C.as.Dec.impl.Op.type.987e49.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %C, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %C.as.Dec.impl.Op.type.987e49.2: type = fn_type @C.as.Dec.impl.Op.loc22_25.2 [concrete]
// CHECK:STDOUT:   %C.as.Dec.impl.Op.26619b.2: %C.as.Dec.impl.Op.type.987e49.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.475: type = fn_type_with_self_type %Dec.Op.type, %Dec.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %impl.elem0: %.475 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%C.as.Dec.impl.Op.26619b.2]
// CHECK:STDOUT:   %bound_method.loc28: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %Op.ref: %C.as.Dec.impl.Op.type.987e49.1 = name_ref Op, @C.as.Dec.impl.%C.as.Dec.impl.Op.decl.loc22_25.1 [concrete = constants.%C.as.Dec.impl.Op.26619b.1]
// CHECK:STDOUT:   %C.as.Dec.impl.Op.bound: <bound method> = bound_method %c.ref, %Op.ref
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %c.ref
// CHECK:STDOUT:   %C.as.Dec.impl.Op.call: init %empty_tuple.type = call %C.as.Dec.impl.Op.bound(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
