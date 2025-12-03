// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_export_name_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_export_name_params.carbon

// --- a.carbon

package Foo library "[[@TEST_NAME]]";

class C1(T:! type);
class C2(T:! type);

// --- fail_b.carbon

package Foo library "[[@TEST_NAME]]";

import library "a";

export C1;

// CHECK:STDERR: fail_b.carbon:[[@LINE+4]]:10: error: `export` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR: export C2(T:! type);
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
export C2(T:! type);

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C1.type: type = generic_class_type @C1 [concrete]
// CHECK:STDOUT:   %C1.generic: %C1.type = struct_value () [concrete]
// CHECK:STDOUT:   %C2.type: type = generic_class_type @C2 [concrete]
// CHECK:STDOUT:   %C2.generic: %C2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: %C1.type = class_decl @C1 [concrete = constants.%C1.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.decl: %C2.type = class_decl @C2 [concrete = constants.%C2.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C1(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C2(%T.loc5_10.2: type) {
// CHECK:STDOUT:   %T.loc5_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(constants.%T) {
// CHECK:STDOUT:   %T.loc5_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1.type: type = generic_class_type @C1 [concrete]
// CHECK:STDOUT:   %C1.generic: %C1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C2.type: type = generic_class_type @C2 [concrete]
// CHECK:STDOUT:   %C2.generic: %C2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Foo.C1: %C1.type = import_ref Foo//a, C1, loaded [concrete = constants.%C1.generic]
// CHECK:STDOUT:   %Foo.C2: %C2.type = import_ref Foo//a, C2, loaded [concrete = constants.%C2.generic]
// CHECK:STDOUT:   %Foo.import_ref.efcd44.1: type = import_ref Foo//a, loc4_10, loaded [symbolic = @C1.%T (constants.%T)]
// CHECK:STDOUT:   %Foo.import_ref.efcd44.2: type = import_ref Foo//a, loc5_10, loaded [symbolic = @C2.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = %C1
// CHECK:STDOUT:     .C2 = %C2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C1: %C1.type = export C1, imports.%Foo.C1 [concrete = constants.%C1.generic]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = constants.%T]
// CHECK:STDOUT:   %C2: %C2.type = export C2, imports.%Foo.C2 [concrete = constants.%C2.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C1(imports.%Foo.import_ref.efcd44.1: type) [from "a.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C2(imports.%Foo.import_ref.efcd44.2: type) [from "a.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
