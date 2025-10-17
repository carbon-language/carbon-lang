// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=ignore
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/basic.carbon

constraint Declared {}

constraint ForwardDeclared;

constraint ForwardDeclared {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Self.85eee6.1: %type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.85eee6.2: %type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Declared = %Declared.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declared.decl: type = constraint_decl @Declared [concrete = constants.%type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc16: type = constraint_decl @ForwardDeclared [concrete = constants.%type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc18: type = constraint_decl @ForwardDeclared [concrete = constants.%type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Declared {
// CHECK:STDOUT:   %Self: %type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.85eee6.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @ForwardDeclared {
// CHECK:STDOUT:   %Self: %type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.85eee6.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
