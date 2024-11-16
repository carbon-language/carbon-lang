// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/fail_export_name_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/fail_export_name_params.carbon

// --- a.carbon

package Foo library "[[@TEST_NAME]]";

class C1(T:! type);
class C2(T:! type);

// --- fail_b.carbon

package Foo library "[[@TEST_NAME]]";

import library "a";

export C1;

// CHECK:STDERR: fail_b.carbon:[[@LINE+3]]:10: error: `export` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR: export C2(T:! type);
// CHECK:STDERR:          ^~~~~~~~~~
export C2(T:! type);

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C1.type: type = generic_class_type @C1 [template]
// CHECK:STDOUT:   %C1.1: %C1.type = struct_value () [template]
// CHECK:STDOUT:   %C2.type: type = generic_class_type @C2 [template]
// CHECK:STDOUT:   %C2.1: %C2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: %C1.type = class_decl @C1 [template = constants.%C1.1] {
// CHECK:STDOUT:     %T.patt.loc4_10.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_10.1, runtime_param<invalid> [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_10.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.decl: %C2.type = class_decl @C2 [template = constants.%C2.1] {
// CHECK:STDOUT:     %T.patt.loc5_10.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc5_10.1, runtime_param<invalid> [symbolic = %T.patt.loc5_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc5_10.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc5_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C1(%T.loc4_10.1: type) {
// CHECK:STDOUT:   %T.loc4_10.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_10.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C2(%T.loc5_10.1: type) {
// CHECK:STDOUT:   %T.loc5_10.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_10.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_10.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(constants.%T) {
// CHECK:STDOUT:   %T.loc5_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_10.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1.type: type = generic_class_type @C1 [template]
// CHECK:STDOUT:   %C1.1: %C1.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C2.type: type = generic_class_type @C2 [template]
// CHECK:STDOUT:   %C2.1: %C2.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %C1.type = import_ref Foo//a, inst+7, loaded [template = constants.%C1.1]
// CHECK:STDOUT:   %import_ref.2: %C2.type = import_ref Foo//a, inst+18, loaded [template = constants.%C2.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C1 = %C1
// CHECK:STDOUT:     .C2 = %C2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C1: %C1.type = export C1, imports.%import_ref.1 [template = constants.%C1.1]
// CHECK:STDOUT:   %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, %T.param [symbolic = constants.%T]
// CHECK:STDOUT:   %C2: %C2.type = export C2, imports.%import_ref.2 [template = constants.%C2.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C1(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C2(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
