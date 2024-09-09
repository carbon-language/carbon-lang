// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_params.carbon

// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:12: ERROR: `namespace` declaration cannot have parameters.
// CHECK:STDERR: namespace A();
// CHECK:STDERR:            ^~
// CHECK:STDERR:
namespace A();

// Parameters are ignored for error recovery.
fn A.F() {}

// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:12: ERROR: `namespace` declaration cannot have parameters.
// CHECK:STDERR: namespace B(n: i32);
// CHECK:STDERR:            ^~~~~~~~
// CHECK:STDERR:
namespace B(n: i32);

// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:12: ERROR: `namespace` declaration cannot have parameters.
// CHECK:STDERR: namespace C[T:! type](x: T);
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
namespace C[T:! type](x: T);

namespace D;
// CHECK:STDERR: fail_params.carbon:[[@LINE+6]]:4: ERROR: Redeclaration differs because of parameter list.
// CHECK:STDERR: fn D(T:! type).F() {}
// CHECK:STDERR:    ^
// CHECK:STDERR: fail_params.carbon:[[@LINE-4]]:1: Previously declared without parameter list.
// CHECK:STDERR: namespace D;
// CHECK:STDERR: ^~~~~~~~~~~~
fn D(T:! type).F() {}

// CHECK:STDOUT: --- fail_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.2: %.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc24_16.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc24_16.2: type = converted %int.make_type_32, %.loc24_16.1 [template = i32]
// CHECK:STDOUT:   %n.loc24_13.1: i32 = param n, runtime_param<invalid>
// CHECK:STDOUT:   %n.loc24_13.2: i32 = bind_name n, %n.loc24_13.1
// CHECK:STDOUT:   %B: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %T.loc30_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:   %T.loc30_13.2: type = bind_symbolic_name T 0, %T.loc30_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T.loc30_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %x.loc30_23.1: %T = param x, runtime_param<invalid>
// CHECK:STDOUT:   %x.loc30_23.2: %T = bind_name x, %x.loc30_23.1
// CHECK:STDOUT:   %C: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.2] {
// CHECK:STDOUT:     %T.loc39_6.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc39_6.2: type = bind_symbolic_name T 0, %T.loc39_6.1 [symbolic = @.1.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(file.%T.loc39_6.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
