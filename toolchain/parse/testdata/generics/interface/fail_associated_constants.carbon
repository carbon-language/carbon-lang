// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/fail_associated_constants.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/fail_associated_constants.carbon

// --- fail_tuple_pattern.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_tuple_pattern.carbon:[[@LINE+2]]:7: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:
  let (T: type, U: type);
}

// --- fail_tuple_pattern_with_default.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_tuple_pattern_with_default.carbon:[[@LINE+2]]:15: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:
  default let (T: type, U: type) = ({}, {});
}

// --- fail_var_pattern.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_var_pattern.carbon:[[@LINE+2]]:7-9: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:
  let var T: type;
}

// --- fail_var_pattern_with_default.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_var_pattern_with_default.carbon:[[@LINE+2]]:15-17: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:
  default let var T: type = {};
}

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_not_constant.carbon:[[@LINE+2]]:7-13: error: phase keyword is not allowed on an associated constant [PhaseKeywordInAssociatedConstant]
  // CHECK:STDERR:
  let runtime a: {.b: ()};
}

// --- fail_template.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_template.carbon:[[@LINE+2]]:7-14: error: phase keyword is not allowed on an associated constant [PhaseKeywordInAssociatedConstant]
  // CHECK:STDERR:
  let template T: type;
}

// --- fail_generic.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+2]]:7-13: error: phase keyword is not allowed on an associated constant [PhaseKeywordInAssociatedConstant]
  // CHECK:STDERR:
  let generic a: type;
}

// --- fail_underscore.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_underscore.carbon:[[@LINE+2]]:7: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:
  let _: {};
}

// --- fail_missing_colon.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_missing_colon.carbon:[[@LINE+2]]:8: error: expected `:` in associated constant declaration [ExpectedAssociatedConstantColon]
  // CHECK:STDERR:
  let a;
}


// CHECK:STDOUT: - filename: fail_tuple_pattern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"tuple_pattern"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_tuple_pattern_with_default.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"tuple_pattern_with_default"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ ├─DefaultModifier 'default'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_var_pattern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"var_pattern"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_var_pattern_with_default.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"var_pattern_with_default"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ ├─DefaultModifier 'default'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_not_constant.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"not_constant"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_template.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"template"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_generic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"generic"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_underscore.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"underscore"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_colon.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"missing_colon"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
