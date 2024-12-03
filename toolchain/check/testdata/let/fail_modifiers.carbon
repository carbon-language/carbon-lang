// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected let b: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected let b: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: default let c: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default let c: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `final` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: final let d: i32 = 1;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
final let d: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `virtual` not allowed on `let` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: virtual let e: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual let e: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `final` not allowed on declaration with `default` [ModifierNotAllowedWith]
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default final let f: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `default` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default default let g: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:11: error: `private` not allowed on declaration with `protected` [ModifierNotAllowedWith]
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected private let h: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+10]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: error: `protected` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: note: `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
protected protected let i: i32 = 1;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:     .e = @__global_init.%e
// CHECK:STDOUT:     .f = @__global_init.%f
// CHECK:STDOUT:     .g = @__global_init.%g
// CHECK:STDOUT:     .h = @__global_init.%h
// CHECK:STDOUT:     .i = @__global_init.%i
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_18.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_18.2: type = converted %int.make_type_signed.loc15, %.loc15_18.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%int_32.loc21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_16.1: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_16.2: type = converted %int.make_type_signed.loc21, %.loc21_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc27: init type = call constants.%Int(%int_32.loc27) [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_14.1: type = value_of_initializer %int.make_type_signed.loc27 [template = constants.%i32]
// CHECK:STDOUT:   %.loc27_14.2: type = converted %int.make_type_signed.loc27, %.loc27_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc33: init type = call constants.%Int(%int_32.loc33) [template = constants.%i32]
// CHECK:STDOUT:   %.loc33_16.1: type = value_of_initializer %int.make_type_signed.loc33 [template = constants.%i32]
// CHECK:STDOUT:   %.loc33_16.2: type = converted %int.make_type_signed.loc33, %.loc33_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc46: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc46: init type = call constants.%Int(%int_32.loc46) [template = constants.%i32]
// CHECK:STDOUT:   %.loc46_22.1: type = value_of_initializer %int.make_type_signed.loc46 [template = constants.%i32]
// CHECK:STDOUT:   %.loc46_22.2: type = converted %int.make_type_signed.loc46, %.loc46_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc59: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc59: init type = call constants.%Int(%int_32.loc59) [template = constants.%i32]
// CHECK:STDOUT:   %.loc59_24.1: type = value_of_initializer %int.make_type_signed.loc59 [template = constants.%i32]
// CHECK:STDOUT:   %.loc59_24.2: type = converted %int.make_type_signed.loc59, %.loc59_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc72: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc72: init type = call constants.%Int(%int_32.loc72) [template = constants.%i32]
// CHECK:STDOUT:   %.loc72_26.1: type = value_of_initializer %int.make_type_signed.loc72 [template = constants.%i32]
// CHECK:STDOUT:   %.loc72_26.2: type = converted %int.make_type_signed.loc72, %.loc72_26.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc84: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc84: init type = call constants.%Int(%int_32.loc84) [template = constants.%i32]
// CHECK:STDOUT:   %.loc84_28.1: type = value_of_initializer %int.make_type_signed.loc84 [template = constants.%i32]
// CHECK:STDOUT:   %.loc84_28.2: type = converted %int.make_type_signed.loc84, %.loc84_28.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %Convert.specific_fn.loc15(%int_1.loc15) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_25.1: %i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_25.2: %i32 = converted %int_1.loc15, %.loc15_25.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc15_25.2
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc21: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc21: <specific function> = specific_function %Convert.bound.loc21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %Convert.specific_fn.loc21(%int_1.loc21) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc21_23.1: %i32 = value_of_initializer %int.convert_checked.loc21 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc21_23.2: %i32 = converted %int_1.loc21, %.loc21_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc21_23.2
// CHECK:STDOUT:   %int_1.loc27: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc27: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc27: <bound method> = bound_method %int_1.loc27, %impl.elem0.loc27 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc27: <specific function> = specific_function %Convert.bound.loc27, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %Convert.specific_fn.loc27(%int_1.loc27) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc27_21.1: %i32 = value_of_initializer %int.convert_checked.loc27 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc27_21.2: %i32 = converted %int_1.loc27, %.loc27_21.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %d: %i32 = bind_name d, %.loc27_21.2
// CHECK:STDOUT:   %int_1.loc33: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc33: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc33: <bound method> = bound_method %int_1.loc33, %impl.elem0.loc33 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc33: <specific function> = specific_function %Convert.bound.loc33, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc33: init %i32 = call %Convert.specific_fn.loc33(%int_1.loc33) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc33_23.1: %i32 = value_of_initializer %int.convert_checked.loc33 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc33_23.2: %i32 = converted %int_1.loc33, %.loc33_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %e: %i32 = bind_name e, %.loc33_23.2
// CHECK:STDOUT:   %int_1.loc46: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc46: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc46: <bound method> = bound_method %int_1.loc46, %impl.elem0.loc46 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc46: <specific function> = specific_function %Convert.bound.loc46, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc46: init %i32 = call %Convert.specific_fn.loc46(%int_1.loc46) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc46_29.1: %i32 = value_of_initializer %int.convert_checked.loc46 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc46_29.2: %i32 = converted %int_1.loc46, %.loc46_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %f: %i32 = bind_name f, %.loc46_29.2
// CHECK:STDOUT:   %int_1.loc59: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc59: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc59: <bound method> = bound_method %int_1.loc59, %impl.elem0.loc59 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc59: <specific function> = specific_function %Convert.bound.loc59, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc59: init %i32 = call %Convert.specific_fn.loc59(%int_1.loc59) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc59_31.1: %i32 = value_of_initializer %int.convert_checked.loc59 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc59_31.2: %i32 = converted %int_1.loc59, %.loc59_31.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %g: %i32 = bind_name g, %.loc59_31.2
// CHECK:STDOUT:   %int_1.loc72: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc72: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc72: <bound method> = bound_method %int_1.loc72, %impl.elem0.loc72 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc72: <specific function> = specific_function %Convert.bound.loc72, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc72: init %i32 = call %Convert.specific_fn.loc72(%int_1.loc72) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc72_33.1: %i32 = value_of_initializer %int.convert_checked.loc72 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc72_33.2: %i32 = converted %int_1.loc72, %.loc72_33.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %h: %i32 = bind_name h, %.loc72_33.2
// CHECK:STDOUT:   %int_1.loc84: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc84: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc84: <bound method> = bound_method %int_1.loc84, %impl.elem0.loc84 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc84: <specific function> = specific_function %Convert.bound.loc84, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc84: init %i32 = call %Convert.specific_fn.loc84(%int_1.loc84) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc84_35.1: %i32 = value_of_initializer %int.convert_checked.loc84 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc84_35.2: %i32 = converted %int_1.loc84, %.loc84_35.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %i: %i32 = bind_name i, %.loc84_35.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
