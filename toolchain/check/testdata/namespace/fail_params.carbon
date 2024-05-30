// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_params.carbon

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

// CHECK:STDERR: fail_params.carbon:[[@LINE+3]]:12: ERROR: `namespace` declaration cannot have parameters.
// CHECK:STDERR: namespace C[T:! type](x: T);
// CHECK:STDERR:            ^~~~~~~~~~
namespace C[T:! type](x: T);

// CHECK:STDOUT: --- fail_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.2: Int32 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:   %import_ref: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc24_16.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc24_16.2: type = converted %int.make_type_32, %.loc24_16.1 [template = i32]
// CHECK:STDOUT:   %n.loc24_13.1: i32 = param n
// CHECK:STDOUT:   %n.loc24_13.2: i32 = bind_name n, %n.loc24_13.1
// CHECK:STDOUT:   %B: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %T.loc29_13.1: type = param T
// CHECK:STDOUT:   %T.loc29_13.2: type = bind_symbolic_name T 0, %T.loc29_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T.loc29_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %x.loc29_23.1: T = param x
// CHECK:STDOUT:   %x.loc29_23.2: T = bind_name x, %x.loc29_23.1
// CHECK:STDOUT:   %C: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
