// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/redeclare.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/redeclare.carbon

// --- redeclare.carbon

library "redeclare";

fn F(T:! type) -> T*;

fn F(T:! type) -> T* {
  return F(T);
}

// --- fail_different_return_type.carbon

library "fail_different_return_type";

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_different_return_type.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because return type is `U*`.
// CHECK:STDERR: fn F(T:! type, U:! type) -> U* {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_different_return_type.carbon:[[@LINE-5]]:1: Previously declared with return type `T*`.
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! type, U:! type) -> U* {
  // TODO: This diagnostic is not very good. We should check the arity in a
  // function call before, or as part of, deduction.
  // CHECK:STDERR: fail_different_return_type.carbon:[[@LINE+7]]:10: ERROR: Cannot deduce value for generic parameter `T`.
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_different_return_type.carbon:[[@LINE-15]]:1: While deducing parameters of generic declared here.
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(T);
}

// --- fail_reorder.carbon

library "fail_reorder";

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_reorder.carbon:[[@LINE+7]]:6: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: fn F(U:! type, T:! type) -> T* {
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_reorder.carbon:[[@LINE-5]]:6: Previous declaration's corresponding parameter here.
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(U:! type, T:! type) -> T* {
  // CHECK:STDERR: fail_reorder.carbon:[[@LINE+7]]:10: ERROR: Cannot deduce value for generic parameter `T`.
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_reorder.carbon:[[@LINE-13]]:1: While deducing parameters of generic declared here.
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(T);
}

// --- fail_rename.carbon

library "fail_rename";

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_rename.carbon:[[@LINE+7]]:6: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: fn F(U:! type, T:! type) -> U* {
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_rename.carbon:[[@LINE-5]]:6: Previous declaration's corresponding parameter here.
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(U:! type, T:! type) -> U* {
  // CHECK:STDERR: fail_rename.carbon:[[@LINE+6]]:10: ERROR: Cannot deduce value for generic parameter `T`.
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_rename.carbon:[[@LINE-13]]:1: While deducing parameters of generic declared here.
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return F(T);
}

// CHECK:STDOUT: --- redeclare.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc4: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc4_7: type = binding_pattern T 0, %T.loc4_6.2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, %T.loc4_6.2 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_20: type = ptr_type %T [symbolic = @F.%.1 (constants.%.1)]
// CHECK:STDOUT:     %return.var.loc4: ref @F.%.1 (%.1) = var <return slot>
// CHECK:STDOUT:     %T.loc4_6.2: type = bind_symbolic_name T 0, %T.loc4_6.3 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_6.3: type = param T
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc6: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc6_7: type = binding_pattern T 0, @F.%T.loc6
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6: type = name_ref T, @F.%T.loc6 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc6_20: type = ptr_type %T [symbolic = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref %.1 = var <return slot>
// CHECK:STDOUT:     @F.%T.loc6: type = bind_symbolic_name T 0, %T.loc6_6.3 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.loc6_6.3: type = param T
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(file.%T.loc4_6.2: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %.1: type = ptr_type @F.%T.1 (%T) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc6: type) -> %.1 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl.loc4 [template = constants.%F]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %F.call: init @F.%.1 (%.1) = call %F.ref(%T.ref)
// CHECK:STDOUT:     %.loc7_14.1: @F.%.1 (%.1) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc7_14.2: @F.%.1 (%.1) = converted %F.call, %.loc7_14.1
// CHECK:STDOUT:     return %.loc7_14.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_different_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc4_7: type = binding_pattern T 0, @F.%T.loc4
// CHECK:STDOUT:     %.loc4_17: type = binding_pattern U 1, @F.%U.loc4
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @F.%T.loc4 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_30: type = ptr_type %T [symbolic = @F.%.1 (constants.%.1)]
// CHECK:STDOUT:     @F.%return: ref @F.%.1 (%.1) = var <return slot>
// CHECK:STDOUT:     @F.%T.loc4: type = bind_symbolic_name T 0, %T.loc4_6.3 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_6.3: type = param T
// CHECK:STDOUT:     @F.%U.loc4: type = bind_symbolic_name U 1, %U.loc4_16.3 [symbolic = @F.%U.1 (constants.%U)]
// CHECK:STDOUT:     %U.loc4_16.3: type = param U
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %.loc13_7: type = binding_pattern T 0, @.1.%T.loc13
// CHECK:STDOUT:     %.loc13_17: type = binding_pattern U 1, @.1.%U.loc13
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, @.1.%U.loc13 [symbolic = @.1.%U.1 (constants.%U)]
// CHECK:STDOUT:     %.loc13_30: type = ptr_type %U [symbolic = @.1.%.1 (constants.%.3)]
// CHECK:STDOUT:     @.1.%return: ref @.1.%.1 (%.3) = var <return slot>
// CHECK:STDOUT:     @.1.%T.loc13: type = bind_symbolic_name T 0, %T.loc13_6.3 [symbolic = @.1.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc13_6.3: type = param T
// CHECK:STDOUT:     @.1.%U.loc13: type = bind_symbolic_name U 1, %U.loc13_16.3 [symbolic = @.1.%U.1 (constants.%U)]
// CHECK:STDOUT:     %U.loc13_16.3: type = param U
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4: type, %U.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   %.1: type = ptr_type @F.%T.1 (%T) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc4: type, %U.loc4: type) -> @F.%.1 (%.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%T.loc13: type, %U.loc13: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   %.1: type = ptr_type @.1.%U.1 (%U) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc13: type, %U.loc13: type) -> @.1.%.1 (%.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc13 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_reorder.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T.1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %U.2: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %T.2: type = bind_symbolic_name T 1 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %T.2 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc4_7: type = binding_pattern T 0, @F.%T.loc4
// CHECK:STDOUT:     %.loc4_17: type = binding_pattern U 1, @F.%U.loc4
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, @F.%T.loc4 [symbolic = @F.%T.1 (constants.%T.1)]
// CHECK:STDOUT:     %.loc4_30: type = ptr_type %T.1 [symbolic = @F.%.1 (constants.%.1)]
// CHECK:STDOUT:     @F.%return: ref @F.%.1 (%.1) = var <return slot>
// CHECK:STDOUT:     @F.%T.loc4: type = bind_symbolic_name T 0, %T.loc4_6.3 [symbolic = @F.%T.1 (constants.%T.1)]
// CHECK:STDOUT:     %T.loc4_6.3: type = param T
// CHECK:STDOUT:     @F.%U.loc4: type = bind_symbolic_name U 1, %U.loc4_16.3 [symbolic = @F.%U.1 (constants.%U.1)]
// CHECK:STDOUT:     %U.loc4_16.3: type = param U
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %.loc13_7: type = binding_pattern U 0, @.1.%U.loc13
// CHECK:STDOUT:     %.loc13_17: type = binding_pattern T 1, @.1.%T.loc13
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, @.1.%T.loc13 [symbolic = @.1.%T.1 (constants.%T.2)]
// CHECK:STDOUT:     %.loc13_30: type = ptr_type %T.2 [symbolic = @.1.%.1 (constants.%.3)]
// CHECK:STDOUT:     @.1.%return: ref @.1.%.1 (%.3) = var <return slot>
// CHECK:STDOUT:     @.1.%U.loc13: type = bind_symbolic_name U 0, %U.loc13_6.3 [symbolic = @.1.%U.1 (constants.%U.2)]
// CHECK:STDOUT:     %U.loc13_6.3: type = param U
// CHECK:STDOUT:     @.1.%T.loc13: type = bind_symbolic_name T 1, %T.loc13_16.3 [symbolic = @.1.%T.1 (constants.%T.2)]
// CHECK:STDOUT:     %T.loc13_16.3: type = param T
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4: type, %U.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U.1)]
// CHECK:STDOUT:   %.1: type = ptr_type @F.%T.1 (%T.1) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc4: type, %U.loc4: type) -> @F.%.1 (%.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%U.loc13: type, %T.loc13: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 0 [symbolic = %U.1 (constants.%U.2)]
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 1 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @.1.%T.1 (%T.2) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc13: type, %T.loc13: type) -> @.1.%.1 (%.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc13 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.1, constants.%U.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %U.1 => constants.%U.1
// CHECK:STDOUT:   %.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%U.2, constants.%T.2) {
// CHECK:STDOUT:   %U.1 => constants.%U.2
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_rename.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T.1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %U.2: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %T.2: type = bind_symbolic_name T 1 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %U.2 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc4_7: type = binding_pattern T 0, @F.%T.loc4
// CHECK:STDOUT:     %.loc4_17: type = binding_pattern U 1, @F.%U.loc4
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @F.%T.loc4 [symbolic = @F.%T.1 (constants.%T.1)]
// CHECK:STDOUT:     %.loc4_30: type = ptr_type %T.1 [symbolic = @F.%.1 (constants.%.1)]
// CHECK:STDOUT:     @F.%return: ref @F.%.1 (%.1) = var <return slot>
// CHECK:STDOUT:     @F.%T.loc4: type = bind_symbolic_name T 0, %T.loc4_6.3 [symbolic = @F.%T.1 (constants.%T.1)]
// CHECK:STDOUT:     %T.loc4_6.3: type = param T
// CHECK:STDOUT:     @F.%U.loc4: type = bind_symbolic_name U 1, %U.loc4_16.3 [symbolic = @F.%U.1 (constants.%U.1)]
// CHECK:STDOUT:     %U.loc4_16.3: type = param U
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %.loc13_7: type = binding_pattern U 0, @.1.%U.loc13
// CHECK:STDOUT:     %.loc13_17: type = binding_pattern T 1, @.1.%T.loc13
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, @.1.%U.loc13 [symbolic = @.1.%U.1 (constants.%U.2)]
// CHECK:STDOUT:     %.loc13_30: type = ptr_type %U.2 [symbolic = @.1.%.1 (constants.%.3)]
// CHECK:STDOUT:     @.1.%return: ref @.1.%.1 (%.3) = var <return slot>
// CHECK:STDOUT:     @.1.%U.loc13: type = bind_symbolic_name U 0, %U.loc13_6.3 [symbolic = @.1.%U.1 (constants.%U.2)]
// CHECK:STDOUT:     %U.loc13_6.3: type = param U
// CHECK:STDOUT:     @.1.%T.loc13: type = bind_symbolic_name T 1, %T.loc13_16.3 [symbolic = @.1.%T.1 (constants.%T.2)]
// CHECK:STDOUT:     %T.loc13_16.3: type = param T
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4: type, %U.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U.1)]
// CHECK:STDOUT:   %.1: type = ptr_type @F.%T.1 (%T.1) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc4: type, %U.loc4: type) -> @F.%.1 (%.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%U.loc13: type, %T.loc13: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 0 [symbolic = %U.1 (constants.%U.2)]
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 1 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @.1.%U.1 (%U.2) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.loc13: type, %T.loc13: type) -> @.1.%.1 (%.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc13 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.1, constants.%U.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT:   %U.1 => constants.%U.1
// CHECK:STDOUT:   %.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%U.2, constants.%T.2) {
// CHECK:STDOUT:   %U.1 => constants.%U.2
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
