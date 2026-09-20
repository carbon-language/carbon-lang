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
// CHECK:STDERR: fail_and.carbon:[[@LINE+2]]:13-25: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var and_: F(true and true);

// --- fail_and_val.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_and_val.carbon:[[@LINE+4]]:21-28: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_and_val.carbon:[[@LINE+2]]:21-33: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var and_val: bool = true and true;

// --- fail_or.carbon

library "[[@TEST_NAME]]";

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// CHECK:STDERR: fail_or.carbon:[[@LINE+2]]:12-23: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var or_: F(true or true);

// --- fail_or_val.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_or_val.carbon:[[@LINE+4]]:20-26: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_or_val.carbon:[[@LINE+2]]:20-31: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var or_val: bool = true or true;

// --- fail_nested.carbon

library "[[@TEST_NAME]]";

base class C(B: bool) {}

fn F() {
  class B {
    // CHECK:STDERR: fail_nested.carbon:[[@LINE+2]]:20-32: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
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
// CHECK:STDOUT:   %Copy.impl_witness.c99: <witness> = impl_witness <unexpected>.inst{{[0-9A-F]+}} [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.c99) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ac1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.224: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ac1, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: bool) -> out %return.param: type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, <unexpected>.inst{{[0-9A-F]+}}.loc5_7
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64 [concrete = constants.%f64]
// CHECK:STDOUT:   br !if.expr.result(%f64)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc6: type = block_arg !if.expr.result
// CHECK:STDOUT:   %impl.elem0: %.224 = impl_witness_access constants.%Copy.impl_witness.c99, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc6, %impl.elem0
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc6)
// CHECK:STDOUT:   return %type.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_and_val.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c99: <witness> = impl_witness <unexpected>.inst{{[0-9A-F]+}} [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.c99) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ac1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.224: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ac1, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: bool) -> out %return.param: type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, <unexpected>.inst{{[0-9A-F]+}}.loc5_7
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64 [concrete = constants.%f64]
// CHECK:STDOUT:   br !if.expr.result(%f64)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc6: type = block_arg !if.expr.result
// CHECK:STDOUT:   %impl.elem0: %.224 = impl_witness_access constants.%Copy.impl_witness.c99, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc6, %impl.elem0
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc6)
// CHECK:STDOUT:   return %type.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_or_val.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %B.patt: %pattern_type = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %B.a76: bool = symbolic_binding B, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%B.a76) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B.c72: type = class_type @B [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(<unexpected>.inst{{[0-9A-F]+}}.loc4_15: bool) {
// CHECK:STDOUT:   %B.patt.loc4_15.2: %pattern_type = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc4_15.2 (constants.%B.patt)]
// CHECK:STDOUT:   %B: bool = symbolic_binding B, 0 [symbolic = %B (constants.%B.a76)]
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
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, <unexpected>.inst{{[0-9A-F]+}}.loc4_23 [concrete = constants.%C.generic]
// CHECK:STDOUT:   %true.loc10_20: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc10: bool = not %true.loc10_20 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc10_25: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc10 br !or.rhs else br !or.result(%true.loc10_25)
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B.c72
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%B.a76) {
// CHECK:STDOUT:   %B.patt.loc4_15.2 => constants.%B.patt
// CHECK:STDOUT:   %B => constants.%B.a76
// CHECK:STDOUT: }
// CHECK:STDOUT:
