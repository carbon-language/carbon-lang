// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/missing_prelude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/missing_prelude.carbon

// --- fail_missing_prelude.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_missing_prelude.carbon:[[@LINE+4]]:8: error: `Core.Int` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: var n: i32;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
var n: i32;

// --- prelude_empty.carbon

package Core library "[[@TEST_NAME]]";

// --- fail_missing_prelude_member.carbon

library "[[@TEST_NAME]]";

import Core library "prelude_empty";

// CHECK:STDERR: fail_missing_prelude_member.carbon:[[@LINE+4]]:8: error: name `Core.Int` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: var n: i32;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
var n: i32;

// --- prelude_fake_int.carbon

package Core library "[[@TEST_NAME]]";

fn Int[T:! type](N:! T) -> {} { return {}; }

// --- use_fake_int.carbon

library "[[@TEST_NAME]]";

import Core library "prelude_fake_int";

var n: {} = i32;

// --- prelude_use_in_prelude.carbon

package Core library "[[@TEST_NAME]]";

// Core is not an imported package here.

// Note that we are intentionally using a different type here from other cases
// to test that this function is called, not the one from prelude_fake_int.
fn Int[T:! type](N:! T) -> () { return (); }

var n: () = i32;

// --- prelude_as_namespace.carbon

library "[[@TEST_NAME]]";

// TODO: Decide whether we want to accept this.
namespace Core;

fn Core.Int[T:! type](N:! T) -> {} { return {}; }

var n: {} = i32;

// --- fail_prelude_as_class.carbon

library "[[@TEST_NAME]]";

// TODO: Decide whether we want to accept this.
class Core {
  fn Int[T:! type](N:! T) -> {} { return {}; }
}

// CHECK:STDERR: fail_prelude_as_class.carbon:[[@LINE+3]]:13: error: `Core.Int` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: var n: {} = i32;
// CHECK:STDERR:             ^~~
var n: {} = i32;

// CHECK:STDOUT: --- fail_missing_prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_prelude_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude_empty
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_fake_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %T.patt.loc4_8.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_8.1, runtime_param<invalid> [symbolic = %T.patt.loc4_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc4_18.1: @Int.%T.loc4_8.2 (%T) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_18.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: @Int.%T.loc4_8.2 (%T) = value_param_pattern %N.patt.loc4_18.1, runtime_param<invalid> [symbolic = %N.patt.loc4_18.2 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_8.1 [symbolic = %T.loc4_8.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_29.2: type = converted %.loc4_29.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_8.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_8.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: @Int.%T.loc4_8.2 (%T) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_18.1: @Int.%T.loc4_8.2 (%T) = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_18.2 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Int(%T.loc4_8.1: type, %N.loc4_18.1: @Int.%T.loc4_8.2 (%T)) {
// CHECK:STDOUT:   %T.loc4_8.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_8.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_18.2: %T = bind_symbolic_name N, 1 [symbolic = %N.loc4_18.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc4_18.2: %T = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_18.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%N.param_patt: @Int.%T.loc4_8.2 (%T)) -> %empty_struct_type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc4_41: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc4_42: %empty_struct_type = converted %.loc4_41, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:     return %.loc4_42
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc4_8.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_8.2 => constants.%T
// CHECK:STDOUT:   %N.loc4_18.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc4_18.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_fake_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %T = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Int.specific_fn: <specific function> = specific_function %Int, @Int(Core.IntLiteral, %int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude_fake_int
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc6_9.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %n.var: ref %empty_struct_type = var n
// CHECK:STDOUT:   %n: ref %empty_struct_type = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Int(constants.%T: type, constants.%N: %T) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: %T = symbolic_binding_pattern N, 1 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%N.param_patt: @Int.%T (%T)) -> %empty_struct_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %Int.specific_fn: <specific function> = specific_function constants.%Int, @Int(Core.IntLiteral, constants.%int_32) [template = constants.%Int.specific_fn]
// CHECK:STDOUT:   %Int.call: init %empty_struct_type = call %Int.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %Int.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(Core.IntLiteral, constants.%int_32) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %T.patt => Core.IntLiteral
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %N.patt => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_use_in_prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.specific_fn: <specific function> = specific_function %Int, @Int(Core.IntLiteral, %int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %T.patt.loc8_8.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_8.1, runtime_param<invalid> [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc8_18.1: @Int.%T.loc8_8.2 (%T) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc8_18.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: @Int.%T.loc8_8.2 (%T) = value_param_pattern %N.patt.loc8_18.1, runtime_param<invalid> [symbolic = %N.patt.loc8_18.2 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %.loc8_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_29.2: type = converted %.loc8_29.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_8.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: @Int.%T.loc8_8.2 (%T) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc8_18.1: @Int.%T.loc8_8.2 (%T) = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc8_18.2 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %n.var: ref %empty_tuple.type = var n
// CHECK:STDOUT:   %n: ref %empty_tuple.type = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Int(%T.loc8_8.1: type, %N.loc8_18.1: @Int.%T.loc8_8.2 (%T)) {
// CHECK:STDOUT:   %T.loc8_8.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_8.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc8_18.2: %T = bind_symbolic_name N, 1 [symbolic = %N.loc8_18.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc8_18.2: %T = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc8_18.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%N.param_patt: @Int.%T.loc8_8.2 (%T)) -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8_41: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_42: %empty_tuple.type = converted %.loc8_41, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc8_42
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %Int.specific_fn: <specific function> = specific_function constants.%Int, @Int(Core.IntLiteral, constants.%int_32) [template = constants.%Int.specific_fn]
// CHECK:STDOUT:   %Int.call: init %empty_tuple.type = call %Int.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %Int.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %N.loc8_18.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc8_18.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(Core.IntLiteral, constants.%int_32) {
// CHECK:STDOUT:   %T.loc8_8.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc8_8.2 => Core.IntLiteral
// CHECK:STDOUT:   %N.loc8_18.2 => constants.%int_32
// CHECK:STDOUT:   %N.patt.loc8_18.2 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_as_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.specific_fn: <specific function> = specific_function %Int, @Int(Core.IntLiteral, %int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %T.patt.loc7_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_13.1, runtime_param<invalid> [symbolic = %T.patt.loc7_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc7_23.1: @Int.%T.loc7_13.2 (%T) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc7_23.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: @Int.%T.loc7_13.2 (%T) = value_param_pattern %N.patt.loc7_23.1, runtime_param<invalid> [symbolic = %N.patt.loc7_23.2 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_13.1 [symbolic = %T.loc7_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc7_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_34.2: type = converted %.loc7_34.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: @Int.%T.loc7_13.2 (%T) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc7_23.1: @Int.%T.loc7_13.2 (%T) = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc7_23.2 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_9.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_9.2: type = converted %.loc9_9.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %n.var: ref %empty_struct_type = var n
// CHECK:STDOUT:   %n: ref %empty_struct_type = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Int(%T.loc7_13.1: type, %N.loc7_23.1: @Int.%T.loc7_13.2 (%T)) {
// CHECK:STDOUT:   %T.loc7_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc7_23.2: %T = bind_symbolic_name N, 1 [symbolic = %N.loc7_23.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc7_23.2: %T = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc7_23.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%N.param_patt: @Int.%T.loc7_13.2 (%T)) -> %empty_struct_type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_46: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_47: %empty_struct_type = converted %.loc7_46, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:     return %.loc7_47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %Int.specific_fn: <specific function> = specific_function constants.%Int, @Int(Core.IntLiteral, constants.%int_32) [template = constants.%Int.specific_fn]
// CHECK:STDOUT:   %Int.call: init %empty_struct_type = call %Int.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %Int.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc7_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_13.2 => constants.%T
// CHECK:STDOUT:   %N.loc7_23.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc7_23.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(Core.IntLiteral, constants.%int_32) {
// CHECK:STDOUT:   %T.loc7_13.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc7_13.2 => Core.IntLiteral
// CHECK:STDOUT:   %N.loc7_23.2 => constants.%int_32
// CHECK:STDOUT:   %N.patt.loc7_23.2 => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_prelude_as_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Core: type = class_type @Core [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.decl: type = class_decl @Core [template = constants.%Core] {} {}
// CHECK:STDOUT:   %.loc12_9.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %n.var: ref %empty_struct_type = var n
// CHECK:STDOUT:   %n: ref %empty_struct_type = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Core {
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %T.patt.loc6_10.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_10.1 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_10.2, runtime_param<invalid> [symbolic = %T.patt.loc6_10.1 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc6_20.2: @Int.%T.loc6_10.1 (%T) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_20.1 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: @Int.%T.loc6_10.1 (%T) = value_param_pattern %N.patt.loc6_20.2, runtime_param<invalid> [symbolic = %N.patt.loc6_20.1 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_10.2 [symbolic = %T.loc6_10.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_31.2: type = converted %.loc6_31.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_10.2: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_10.1 (constants.%T)]
// CHECK:STDOUT:     %N.param: @Int.%T.loc6_10.1 (%T) = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_20.2: @Int.%T.loc6_10.1 (%T) = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc6_20.1 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Core
// CHECK:STDOUT:   .Int = %Int.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Int(%T.loc6_10.2: type, %N.loc6_20.2: @Int.%T.loc6_10.1 (%T)) {
// CHECK:STDOUT:   %T.loc6_10.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_10.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_10.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_10.1 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc6_20.1: %T = bind_symbolic_name N, 1 [symbolic = %N.loc6_20.1 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_20.1: %T = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_20.1 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%N.param_patt: @Int.%T.loc6_10.1 (%T)) -> %empty_struct_type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc6_43: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_44: %empty_struct_type = converted %.loc6_43, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:     return %.loc6_44
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   assign file.%n.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6_10.1 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_10.1 => constants.%T
// CHECK:STDOUT:   %N.loc6_20.1 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_20.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
