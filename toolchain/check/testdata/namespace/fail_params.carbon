// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_params.carbon

// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:12: error: `namespace` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR: namespace A();
// CHECK:STDERR:            ^~
// CHECK:STDERR:
namespace A();

// Parameters are ignored for error recovery.
fn A.F() {}

// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:12: error: `namespace` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR: namespace B(n: i32);
// CHECK:STDERR:            ^~~~~~~~
// CHECK:STDERR:
namespace B(n: i32);

// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:12: error: `namespace` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR: namespace C[T:! type](x: T);
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
namespace C[T:! type](x: T);

namespace D;
// CHECK:STDERR: fail_params.carbon:[[@LINE+6]]:4: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fn D(T:! type).F() {}
// CHECK:STDERR:    ^
// CHECK:STDERR: fail_params.carbon:[[@LINE-4]]:1: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR: namespace D;
// CHECK:STDERR: ^~~~~~~~~~~~
fn D(T:! type).F() {}

// CHECK:STDOUT: --- fail_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.2: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc24_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc24_16.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_16.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_16.3: type = converted %int.make_type_signed, %.loc24_16.2 [template = constants.%i32]
// CHECK:STDOUT:   %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:   %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:   %B: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, %T.param [symbolic = constants.%T]
// CHECK:STDOUT:   %x.param: %T = value_param runtime_param0
// CHECK:STDOUT:   %x: %T = bind_name x, %x.param
// CHECK:STDOUT:   %C: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.2] {} {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc39_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc39_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%T.loc39_6.1: type) {
// CHECK:STDOUT:   %T.loc39_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc39_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
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
// CHECK:STDOUT:   %T.loc39_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
