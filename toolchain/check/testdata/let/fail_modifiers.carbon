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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
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
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_18.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_18.2: type = converted %int.make_type_32.loc15, %.loc15_18.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_16.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_16.2: type = converted %int.make_type_32.loc21, %.loc21_16.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc27_14.1: type = value_of_initializer %int.make_type_32.loc27 [template = i32]
// CHECK:STDOUT:   %.loc27_14.2: type = converted %int.make_type_32.loc27, %.loc27_14.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc33_16.1: type = value_of_initializer %int.make_type_32.loc33 [template = i32]
// CHECK:STDOUT:   %.loc33_16.2: type = converted %int.make_type_32.loc33, %.loc33_16.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc46: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc46_22.1: type = value_of_initializer %int.make_type_32.loc46 [template = i32]
// CHECK:STDOUT:   %.loc46_22.2: type = converted %int.make_type_32.loc46, %.loc46_22.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc59: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc59_24.1: type = value_of_initializer %int.make_type_32.loc59 [template = i32]
// CHECK:STDOUT:   %.loc59_24.2: type = converted %int.make_type_32.loc59, %.loc59_24.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc72: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc72_26.1: type = value_of_initializer %int.make_type_32.loc72 [template = i32]
// CHECK:STDOUT:   %.loc72_26.2: type = converted %int.make_type_32.loc72, %.loc72_26.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc84: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc84_28.1: type = value_of_initializer %int.make_type_32.loc84 [template = i32]
// CHECK:STDOUT:   %.loc84_28.2: type = converted %int.make_type_32.loc84, %.loc84_28.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_24: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc15_25.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc15_25.2: <bound method> = bound_method %.loc15_24, %.loc15_25.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc15: init i32 = call %.loc15_25.2(%.loc15_24) [template = constants.%.27]
// CHECK:STDOUT:   %.loc15_25.3: i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%.27]
// CHECK:STDOUT:   %.loc15_25.4: i32 = converted %.loc15_24, %.loc15_25.3 [template = constants.%.27]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc15_25.4
// CHECK:STDOUT:   %.loc21_22: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc21_23.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc21_23.2: <bound method> = bound_method %.loc21_22, %.loc21_23.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc21: init i32 = call %.loc21_23.2(%.loc21_22) [template = constants.%.27]
// CHECK:STDOUT:   %.loc21_23.3: i32 = value_of_initializer %int.convert_checked.loc21 [template = constants.%.27]
// CHECK:STDOUT:   %.loc21_23.4: i32 = converted %.loc21_22, %.loc21_23.3 [template = constants.%.27]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc21_23.4
// CHECK:STDOUT:   %.loc27_20: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc27_21.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc27_21.2: <bound method> = bound_method %.loc27_20, %.loc27_21.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc27: init i32 = call %.loc27_21.2(%.loc27_20) [template = constants.%.27]
// CHECK:STDOUT:   %.loc27_21.3: i32 = value_of_initializer %int.convert_checked.loc27 [template = constants.%.27]
// CHECK:STDOUT:   %.loc27_21.4: i32 = converted %.loc27_20, %.loc27_21.3 [template = constants.%.27]
// CHECK:STDOUT:   %d: i32 = bind_name d, %.loc27_21.4
// CHECK:STDOUT:   %.loc33_22: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc33_23.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc33_23.2: <bound method> = bound_method %.loc33_22, %.loc33_23.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc33: init i32 = call %.loc33_23.2(%.loc33_22) [template = constants.%.27]
// CHECK:STDOUT:   %.loc33_23.3: i32 = value_of_initializer %int.convert_checked.loc33 [template = constants.%.27]
// CHECK:STDOUT:   %.loc33_23.4: i32 = converted %.loc33_22, %.loc33_23.3 [template = constants.%.27]
// CHECK:STDOUT:   %e: i32 = bind_name e, %.loc33_23.4
// CHECK:STDOUT:   %.loc46_28: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc46_29.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc46_29.2: <bound method> = bound_method %.loc46_28, %.loc46_29.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc46: init i32 = call %.loc46_29.2(%.loc46_28) [template = constants.%.27]
// CHECK:STDOUT:   %.loc46_29.3: i32 = value_of_initializer %int.convert_checked.loc46 [template = constants.%.27]
// CHECK:STDOUT:   %.loc46_29.4: i32 = converted %.loc46_28, %.loc46_29.3 [template = constants.%.27]
// CHECK:STDOUT:   %f: i32 = bind_name f, %.loc46_29.4
// CHECK:STDOUT:   %.loc59_30: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc59_31.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc59_31.2: <bound method> = bound_method %.loc59_30, %.loc59_31.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc59: init i32 = call %.loc59_31.2(%.loc59_30) [template = constants.%.27]
// CHECK:STDOUT:   %.loc59_31.3: i32 = value_of_initializer %int.convert_checked.loc59 [template = constants.%.27]
// CHECK:STDOUT:   %.loc59_31.4: i32 = converted %.loc59_30, %.loc59_31.3 [template = constants.%.27]
// CHECK:STDOUT:   %g: i32 = bind_name g, %.loc59_31.4
// CHECK:STDOUT:   %.loc72_32: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc72_33.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc72_33.2: <bound method> = bound_method %.loc72_32, %.loc72_33.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc72: init i32 = call %.loc72_33.2(%.loc72_32) [template = constants.%.27]
// CHECK:STDOUT:   %.loc72_33.3: i32 = value_of_initializer %int.convert_checked.loc72 [template = constants.%.27]
// CHECK:STDOUT:   %.loc72_33.4: i32 = converted %.loc72_32, %.loc72_33.3 [template = constants.%.27]
// CHECK:STDOUT:   %h: i32 = bind_name h, %.loc72_33.4
// CHECK:STDOUT:   %.loc84_34: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc84_35.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc84_35.2: <bound method> = bound_method %.loc84_34, %.loc84_35.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc84: init i32 = call %.loc84_35.2(%.loc84_34) [template = constants.%.27]
// CHECK:STDOUT:   %.loc84_35.3: i32 = value_of_initializer %int.convert_checked.loc84 [template = constants.%.27]
// CHECK:STDOUT:   %.loc84_35.4: i32 = converted %.loc84_34, %.loc84_35.3 [template = constants.%.27]
// CHECK:STDOUT:   %i: i32 = bind_name i, %.loc84_35.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
