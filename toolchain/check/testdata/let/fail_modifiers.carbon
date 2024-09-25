// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members
// CHECK:STDERR: protected let b: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected let b: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `default` not allowed on `let` declaration outside of an interface
// CHECK:STDERR: default let c: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default let c: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `final` not allowed on `let` declaration outside of an interface
// CHECK:STDERR: final let d: i32 = 1;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
final let d: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `virtual` not allowed on `let` declaration
// CHECK:STDERR: virtual let e: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual let e: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: error: `default` not allowed on `let` declaration outside of an interface
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `final` not allowed on declaration with `default`
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `default` previously appeared here
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default final let f: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: error: `default` not allowed on `let` declaration outside of an interface
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `default` repeated on declaration
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `default` previously appeared here
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default default let g: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: error: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:11: error: `private` not allowed on declaration with `protected`
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `protected` previously appeared here
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected private let h: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+10]]:1: error: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: error: `protected` repeated on declaration
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: note: `protected` previously appeared here
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
protected protected let i: i32 = 1;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc15
// CHECK:STDOUT:   %.loc21: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc21
// CHECK:STDOUT:   %.loc27: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %d: i32 = bind_name d, %.loc27
// CHECK:STDOUT:   %.loc33: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %e: i32 = bind_name e, %.loc33
// CHECK:STDOUT:   %.loc46: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %f: i32 = bind_name f, %.loc46
// CHECK:STDOUT:   %.loc59: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %g: i32 = bind_name g, %.loc59
// CHECK:STDOUT:   %.loc72: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %h: i32 = bind_name h, %.loc72
// CHECK:STDOUT:   %.loc84: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %i: i32 = bind_name i, %.loc84
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
