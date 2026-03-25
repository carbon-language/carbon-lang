// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

// CHECK:STDERR: fail_params.carbon:[[@LINE+8]]:16: error: name `Core.Int` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: namespace B(n: i32);
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
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
// CHECK:STDERR: fail_params.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fn D(unused T:! type).F() {}
// CHECK:STDERR:    ^
// CHECK:STDERR: fail_params.carbon:[[@LINE-4]]:1: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR: namespace D;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn D(unused T:! type).F() {}

// CHECK:STDOUT: --- fail_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.6d5: type = fn_type @F.loc22 [concrete]
// CHECK:STDOUT:   %F.e11: %F.type.6d5 = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.67db0b.2: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.117: type = fn_type @F.loc48 [concrete]
// CHECK:STDOUT:   %F.d98: %F.type.117 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc22: %F.type.6d5 = fn_decl @F.loc22 [concrete = constants.%F.e11] {} {}
// CHECK:STDOUT:   %n.param: <error> = value_param call_param0
// CHECK:STDOUT:   %.loc32: type = type_literal <error> [concrete = <error>]
// CHECK:STDOUT:   %n: <error> = value_binding n, <error> [concrete = <error>]
// CHECK:STDOUT:   %B: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %.loc38_17.1: type = splice_block %.loc38_17.2 [concrete = type] {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.loc38_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = constants.%T.67db0b.1]
// CHECK:STDOUT:   %x.param: %T.67db0b.1 = value_param call_param0
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T.67db0b.1]
// CHECK:STDOUT:   %x: %T.67db0b.1 = value_binding x, %x.param
// CHECK:STDOUT:   %C: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %D: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %F.decl.loc48: %F.type.117 = fn_decl @F.loc48 [concrete = constants.%F.d98] {} {
// CHECK:STDOUT:     %.loc48_17.1: type = splice_block %.loc48_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc48_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc48_14.3: type = symbolic_binding T, 0 [symbolic = %T.loc48_14.2 (constants.%T.67db0b.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc22() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc48(%T.loc48_14.3: type) {
// CHECK:STDOUT:   %T.loc48_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc48_14.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %T.loc48_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc48_14.2 (constants.%T.67db0b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc48(constants.%T.67db0b.2) {
// CHECK:STDOUT:   %T.loc48_14.1 => constants.%T.67db0b.2
// CHECK:STDOUT:   %T.loc48_14.2 => constants.%T.67db0b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
