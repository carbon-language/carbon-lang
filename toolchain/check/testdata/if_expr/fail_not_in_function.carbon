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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/fail_not_in_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/fail_not_in_function.carbon

// --- fail_basic.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_basic.carbon:[[@LINE+12]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_basic.carbon:[[@LINE+8]]:22: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_basic.carbon:[[@LINE+4]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let x: i32 = if true then 1 else 0;

// --- fail_types.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_types.carbon:[[@LINE+4]]:8: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: if true then i32 else f64;
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var y: if true then i32 else f64;

// --- fail_in_param.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_in_param.carbon:[[@LINE+4]]:9: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: fn F(a: if true then i32 else f64);
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(a: if true then i32 else f64);

// --- fail_class.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+12]]:10: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:18: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:                  ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:10: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var n: if true then i32 else f64;
}

// --- fail_nested_class.carbon

library "[[@TEST_NAME]]";

base class C(T:! type) {}

fn F() {
  class B {
    // CHECK:STDERR: fail_nested_class.carbon:[[@LINE+4]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
    // CHECK:STDERR:     extend base: C(if true then i32 else i32);
    // CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    extend base: C(if true then i32 else i32);
  }
}

// CHECK:STDOUT: --- fail_basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.7d2: type = ptr_type %C [symbolic]
// CHECK:STDOUT:   %pattern_type.1d2: type = pattern_type %ptr.7d2 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(<unexpected>.inst20.loc4_14: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.1d2) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.1d2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_24.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2) = value_param call_param0
// CHECK:STDOUT:       %.loc4_24.2: type = splice_block %Self.ref [symbolic = %C (constants.%C)] {
// CHECK:STDOUT:         %.loc4_24.3: type = specific_constant constants.%C, @C(constants.%T) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_24.3 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(<unexpected>.inst20.loc4_14: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [symbolic = @C.as.Destroy.impl.%C (constants.%C)]
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%T) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, <unexpected>.inst25.loc4_24 [concrete = constants.%C.generic]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(<unexpected>.inst20.loc4_14: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.7d2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.1d2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %ptr => constants.%ptr.7d2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
