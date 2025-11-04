// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/basic.carbon

// --- definition.carbon
library "[[@TEST_NAME]]";
//@include-in-dumps

constraint Declared {}

// --- forward_declaration.carbon
library "[[@TEST_NAME]]";
//@include-in-dumps

constraint ForwardDeclared;

constraint ForwardDeclared {}

// --- fail_name_conflict.carbon
library "[[@TEST_NAME]]";

class ForwardDeclared;

// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+7]]:12: error: duplicate name `ForwardDeclared` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: constraint ForwardDeclared {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-5]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class ForwardDeclared;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
constraint ForwardDeclared {}

// --- fail_defined_twice.carbon
library "[[@TEST_NAME]]";

constraint Declared {}

// CHECK:STDERR: fail_defined_twice.carbon:[[@LINE+7]]:1: error: redefinition of `constraint Declared` [RedeclRedef]
// CHECK:STDERR: constraint Declared {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_defined_twice.carbon:[[@LINE-5]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: constraint Declared {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
constraint Declared {}

// CHECK:STDOUT: --- definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Declared.type: type = facet_type <@Declared> [concrete]
// CHECK:STDOUT:   %Self: %Declared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Declared = %Declared.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declared.decl: type = constraint_decl @Declared [concrete = constants.%Declared.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @Declared {
// CHECK:STDOUT:   %Self: %Declared.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forward_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclared.decl.loc4: type = constraint_decl @ForwardDeclared [concrete = constants.%ForwardDeclared.type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc6: type = constraint_decl @ForwardDeclared [concrete = constants.%ForwardDeclared.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @ForwardDeclared {
// CHECK:STDOUT:   %Self: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
