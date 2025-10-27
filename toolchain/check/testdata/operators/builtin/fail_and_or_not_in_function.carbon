// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon

// --- fail_and.carbon

library "[[@TEST_NAME]]";

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// TODO: Short-circuit operators should be permitted outside functions.
// CHECK:STDERR: fail_and.carbon:[[@LINE+4]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR:
var and_: F(true and true);

// --- fail_and_val.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_and_val.carbon:[[@LINE+8]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_val: bool = true and true;
// CHECK:STDERR:                     ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_val.carbon:[[@LINE+4]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_val: bool = true and true;
// CHECK:STDERR:                     ^~~~~~~~~~~~~
// CHECK:STDERR:
var and_val: bool = true and true;

// --- fail_or.carbon

library "[[@TEST_NAME]]";

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// CHECK:STDERR: fail_or.carbon:[[@LINE+4]]:12: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~~~~~~
// CHECK:STDERR:
var or_: F(true or true);

// --- fail_or_val.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_or_val.carbon:[[@LINE+8]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_val: bool = true or true;
// CHECK:STDERR:                    ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_or_val.carbon:[[@LINE+4]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_val: bool = true or true;
// CHECK:STDERR:                    ^~~~~~~~~~~~
// CHECK:STDERR:
var or_val: bool = true or true;

// --- fail_nested.carbon

library "[[@TEST_NAME]]";

base class C(B:! bool) {}

fn F() {
  class B {
    // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
    // CHECK:STDERR:     extend base: C(true or false);
    // CHECK:STDERR:                    ^~~~~~~~~~~~~
    // CHECK:STDERR:
    extend base: C(true or false);
  }
}

// CHECK:STDOUT: --- fail_and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.de9: <witness> = impl_witness <unexpected>.inst60000156 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.de9) [concrete]
// CHECK:STDOUT:   %.98f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: bool) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, <unexpected>.inst6000001C.loc5_6
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %f64: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64]
// CHECK:STDOUT:   br !if.expr.result(%f64)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc6: type = block_arg !if.expr.result
// CHECK:STDOUT:   %impl.elem0: %.98f = impl_witness_access constants.%Copy.impl_witness.de9, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc6, %impl.elem0
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc6)
// CHECK:STDOUT:   return %type.as.Copy.impl.Op.call to <unexpected>.inst60000025.loc5_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_and_val.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.de9: <witness> = impl_witness <unexpected>.inst70000156 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.de9) [concrete]
// CHECK:STDOUT:   %.98f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: bool) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, <unexpected>.inst7000001C.loc5_6
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %f64: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64]
// CHECK:STDOUT:   br !if.expr.result(%f64)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc6: type = block_arg !if.expr.result
// CHECK:STDOUT:   %impl.elem0: %.98f = impl_witness_access constants.%Copy.impl_witness.de9, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc6, %impl.elem0
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc6)
// CHECK:STDOUT:   return %type.as.Copy.impl.Op.call to <unexpected>.inst70000025.loc5_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_or_val.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.7dd: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%B.7dd) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B.d97: type = class_type @B [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(<unexpected>.inst6800001F.loc4_14: bool) {
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic = %B (constants.%B.7dd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, <unexpected>.inst68000025.loc4_24 [concrete = constants.%C.generic]
// CHECK:STDOUT:   %true.loc12_20: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc12: bool = not %true.loc12_20 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc12_25: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc12 br !or.rhs else br !or.result(%true.loc12_25)
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B.d97
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%B.7dd) {
// CHECK:STDOUT:   %B => constants.%B.7dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
