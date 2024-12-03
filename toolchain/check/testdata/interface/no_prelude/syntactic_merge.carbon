// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/syntactic_merge.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/syntactic_merge.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a:! C);
interface Foo(a:! C) {}

interface Bar(a:! D);
interface Bar(a:! D) {}

// --- spacing.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo [ ] ( a :! C );
interface Foo[](a:! C) {}

// --- fail_parens.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a:! C);
// CHECK:STDERR: fail_parens.carbon:[[@LINE+7]]:19: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: interface Foo(a:! (C)) {}
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_parens.carbon:[[@LINE-4]]:19: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR:                   ^
// CHECK:STDERR:
interface Foo(a:! (C)) {}

// --- todo_fail_raw_identifier.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a:! C);
interface Foo(a:! r#C) {}

// --- two_file.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a:! C);
interface Bar(a:! D);

// --- fail_todo_two_file.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE+8]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface Foo(a:! C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE-5]]:6: in import [InImport]
// CHECK:STDERR: two_file.carbon:7:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Foo(a:! C) {}
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE+8]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface Bar(a:! D) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE-14]]:6: in import [InImport]
// CHECK:STDERR: two_file.carbon:8:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: interface Bar(a:! D);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Bar(a:! D) {}

// --- fail_name_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a:! C);
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE+7]]:15: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: interface Foo(b:! D) {}
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-4]]:15: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR:
interface Foo(b:! D) {}

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a:! C);
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:19: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: interface Foo(a:! D) {}
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE-4]]:19: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR:                   ^
// CHECK:STDERR:
interface Foo(a:! D) {}

// --- fail_deduced_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo[a:! C]();
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE+7]]:19: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: interface Foo[a:! D]() {}
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE-4]]:19: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: interface Foo[a:! C]();
// CHECK:STDERR:                   ^
// CHECK:STDERR:
interface Foo[a:! D]() {}

// --- alias_two_file.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a:! C);

// --- fail_alias_two_file.impl.carbon

impl library "[[@TEST_NAME]]";

alias D = C;

// TODO: This fails because importing interfaces doesn't work well. It should
// fail due to `C` versus `D`, but may succeed if importing interfaces is fixed
// before syntax matching on imports is supported.
// CHECK:STDERR: fail_alias_two_file.impl.carbon:[[@LINE+8]]:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface Foo(a:! D) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_alias_two_file.impl.carbon:[[@LINE-10]]:6: in import [InImport]
// CHECK:STDERR: alias_two_file.carbon:6:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Foo(a:! D) {}

// --- fail_repeat_const.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a:! const C);
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+10]]:19: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: interface Foo(a:! const (const C)) {}
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+6]]:25: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: interface Foo(a:! const (const C)) {}
// CHECK:STDERR:                         ^
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE-8]]:25: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: interface Foo(a:! const C);
// CHECK:STDERR:                         ^
interface Foo(a:! const (const C)) {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.1: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Foo.type.2: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self.1: %Foo.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.1: type = generic_interface_type @Bar [template]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Bar.type.2: type = facet_type <@Bar, @Bar(%a)> [symbolic]
// CHECK:STDOUT:   %Self.2: %Bar.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc7
// CHECK:STDOUT:     .Bar = %Bar.decl.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.1 = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_15.1, runtime_param<invalid> [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0, %a.param.loc7 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type.1 = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_15.1, runtime_param<invalid> [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc8: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc8: %C = bind_symbolic_name a, 0, %a.param.loc8 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type.1 = interface_decl @Bar [template = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc10_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc10_15.1, runtime_param<invalid> [symbolic = %a.patt.loc10_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc10: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc10_15.1: %C = bind_symbolic_name a, 0, %a.param.loc10 [symbolic = %a.loc10_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type.1 = interface_decl @Bar [template = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc10_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc10_15.1, runtime_param<invalid> [symbolic = %a.patt.loc10_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc11: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc11: %C = bind_symbolic_name a, 0, %a.param.loc11 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc7_15.2)> [symbolic = %Foo.type (constants.%Foo.type.2)]
// CHECK:STDOUT:   %Self.2: %Foo.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%Foo.type (%Foo.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%a.loc10_15.1: %C) {
// CHECK:STDOUT:   %a.loc10_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc10_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc10_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%a.loc10_15.2)> [symbolic = %Bar.type (constants.%Bar.type.2)]
// CHECK:STDOUT:   %Self.2: %Bar.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Bar.%Bar.type (%Bar.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(%a.loc7_15.2) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc10_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc10_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(%a.loc10_15.2) {
// CHECK:STDOUT:   %a.loc10_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc10_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.1: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Foo.type.2: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.1 = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_21.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_21.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6_21.1, runtime_param<invalid> [symbolic = %a.patt.loc6_21.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc6: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_21.1: %C = bind_symbolic_name a, 0, %a.param.loc6 [symbolic = %a.loc6_21.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.1 = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_21.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_21.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6_21.1, runtime_param<invalid> [symbolic = %a.patt.loc6_21.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0, %a.param.loc7 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_21.1: %C) {
// CHECK:STDOUT:   %a.loc6_21.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_21.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_21.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_21.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc6_21.2)> [symbolic = %Foo.type (constants.%Foo.type.2)]
// CHECK:STDOUT:   %Self.2: %Foo.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%Foo.type (%Foo.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_21.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_21.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(%a.loc6_21.2) {
// CHECK:STDOUT:   %a.loc6_21.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_21.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.generic: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = facet_type <@.1, @.1(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6_15.1, runtime_param<invalid> [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type.1 = interface_decl @.1 [template = constants.%.generic] {
// CHECK:STDOUT:     %a.patt.loc14_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc14_15.1, runtime_param<invalid> [symbolic = %a.patt.loc14_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc14_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc14_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_15.1: %C) {
// CHECK:STDOUT:   %a.loc6_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(%a.loc14_15.1: %C) {
// CHECK:STDOUT:   %a.loc14_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc14_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc14_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.1, @.1(%a.loc14_15.2)> [symbolic = %.type (constants.%.type.2)]
// CHECK:STDOUT:   %Self.2: %.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.type (%.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc14_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc14_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(%a.loc14_15.2) {
// CHECK:STDOUT:   %a.loc14_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc14_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.1: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Foo.type.2: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.1 = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6_15.1, runtime_param<invalid> [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc6: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_15.1: %C = bind_symbolic_name a, 0, %a.param.loc6 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.1 = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6_15.1, runtime_param<invalid> [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0, %a.param.loc7 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_15.1: %C) {
// CHECK:STDOUT:   %a.loc6_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc6_15.2)> [symbolic = %Foo.type (constants.%Foo.type.2)]
// CHECK:STDOUT:   %Self.2: %Foo.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%Foo.type (%Foo.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(%a.loc6_15.2) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Bar.type: type = generic_interface_type @Bar [template]
// CHECK:STDOUT:   %Bar.generic: %Bar.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_15.1, runtime_param<invalid> [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = interface_decl @Bar [template = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc8_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc8_15.1, runtime_param<invalid> [symbolic = %a.patt.loc8_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc8_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc8_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%a.loc8_15.1: %C) {
// CHECK:STDOUT:   %a.loc8_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc8_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc8_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc8_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc8_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.generic.1: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = facet_type <@.1, @.1(%a)> [symbolic]
// CHECK:STDOUT:   %Self.1: %.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type: type = generic_interface_type @Bar [template]
// CHECK:STDOUT:   %Bar.generic: %Bar.type = struct_value () [template]
// CHECK:STDOUT:   %.type.3: type = generic_interface_type @.2 [template]
// CHECK:STDOUT:   %.generic.2: %.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.type.4: type = facet_type <@.2, @.2(%a)> [symbolic]
// CHECK:STDOUT:   %Self.2: %.type.4 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//two_file, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//two_file, inst+7, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3: %Foo.type = import_ref Main//two_file, inst+15, loaded [template = constants.%Foo.generic]
// CHECK:STDOUT:   %import_ref.4: %Bar.type = import_ref Main//two_file, inst+26, loaded [template = constants.%Bar.generic]
// CHECK:STDOUT:   %import_ref.5: <witness> = import_ref Main//two_file, inst+4, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//two_file, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .D = imports.%import_ref.2
// CHECK:STDOUT:     .Foo = imports.%import_ref.3
// CHECK:STDOUT:     .Bar = imports.%import_ref.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %.decl.loc12: %.type.1 = interface_decl @.1 [template = constants.%.generic.1] {
// CHECK:STDOUT:     %a.patt.loc12_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc12_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc12_15.1, runtime_param<invalid> [symbolic = %a.patt.loc12_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc12_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc12_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc21: %.type.3 = interface_decl @.2 [template = constants.%.generic.2] {
// CHECK:STDOUT:     %a.patt.loc21_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc21_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc21_15.1, runtime_param<invalid> [symbolic = %a.patt.loc21_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.2 [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc21_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc21_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(constants.%a: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(%a.loc12_15.1: %C) {
// CHECK:STDOUT:   %a.loc12_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc12_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc12_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc12_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.1, @.1(%a.loc12_15.2)> [symbolic = %.type (constants.%.type.2)]
// CHECK:STDOUT:   %Self.2: %.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.type (%.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(constants.%a: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.2(%a.loc21_15.1: %C) {
// CHECK:STDOUT:   %a.loc21_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc21_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc21_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc21_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.2, @.2(%a.loc21_15.2)> [symbolic = %.type (constants.%.type.4)]
// CHECK:STDOUT:   %Self.2: %.type.4 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.2.%.type (%.type.4) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %a.patt => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc12_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc12_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(%a.loc12_15.2) {
// CHECK:STDOUT:   %a.loc12_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc12_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %a.patt => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.2(constants.%a) {
// CHECK:STDOUT:   %a.loc21_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc21_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.2(%a.loc21_15.2) {
// CHECK:STDOUT:   %a.loc21_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc21_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %b: %C = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.generic: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = facet_type <@.1, @.1(%b)> [symbolic]
// CHECK:STDOUT:   %Self: %.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_15.1, runtime_param<invalid> [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type.1 = interface_decl @.1 [template = constants.%.generic] {
// CHECK:STDOUT:     %b.patt.loc15_15.1: %C = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_15.2 (constants.%b.patt)]
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt.loc15_15.1, runtime_param<invalid> [symbolic = %b.patt.loc15_15.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc15_15.1: %C = bind_symbolic_name b, 0, %b.param [symbolic = %b.loc15_15.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(%b.loc15_15.1: %C) {
// CHECK:STDOUT:   %b.loc15_15.2: %C = bind_symbolic_name b, 0 [symbolic = %b.loc15_15.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc15_15.2: %C = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_15.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.1, @.1(%b.loc15_15.2)> [symbolic = %.type (constants.%.type.2)]
// CHECK:STDOUT:   %Self.2: %.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.type (%.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%b) {
// CHECK:STDOUT:   %b.loc15_15.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc15_15.2 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(%b.loc15_15.2) {
// CHECK:STDOUT:   %b.loc15_15.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc15_15.2 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.generic: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = facet_type <@.1, @.1(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_15.1, runtime_param<invalid> [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type.1 = interface_decl @.1 [template = constants.%.generic] {
// CHECK:STDOUT:     %a.patt.loc15_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc15_15.1, runtime_param<invalid> [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc15_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc15_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(%a.loc15_15.1: %C) {
// CHECK:STDOUT:   %a.loc15_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc15_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.1, @.1(%a.loc15_15.2)> [symbolic = %.type (constants.%.type.2)]
// CHECK:STDOUT:   %Self.2: %.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.type (%.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc15_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(%a.loc15_15.2) {
// CHECK:STDOUT:   %a.loc15_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduced_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.generic: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = facet_type <@.1, @.1(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_15.1, runtime_param<invalid> [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type.1 = interface_decl @.1 [template = constants.%.generic] {
// CHECK:STDOUT:     %a.patt.loc15_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc15_15.1, runtime_param<invalid> [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc15_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc15_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(%a.loc15_15.1: %C) {
// CHECK:STDOUT:   %a.loc15_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc15_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.1, @.1(%a.loc15_15.2)> [symbolic = %.type (constants.%.type.2)]
// CHECK:STDOUT:   %Self.2: %.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.type (%.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc15_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(%a.loc15_15.2) {
// CHECK:STDOUT:   %a.loc15_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6_15.1, runtime_param<invalid> [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_15.1: %C) {
// CHECK:STDOUT:   %a.loc6_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.generic: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = facet_type <@.1, @.1(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//alias_two_file, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: %Foo.type = import_ref Main//alias_two_file, inst+13, loaded [template = constants.%Foo.generic]
// CHECK:STDOUT:   %import_ref.3: <witness> = import_ref Main//alias_two_file, inst+4, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//alias_two_file, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .Foo = imports.%import_ref.2
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %.decl: %.type.1 = interface_decl @.1 [template = constants.%.generic] {
// CHECK:STDOUT:     %a.patt.loc17_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc17_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc17_15.1, runtime_param<invalid> [symbolic = %a.patt.loc17_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc17_15.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc17_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(constants.%a: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(%a.loc17_15.1: %C) {
// CHECK:STDOUT:   %a.loc17_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc17_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc17_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc17_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.1, @.1(%a.loc17_15.2)> [symbolic = %.type (constants.%.type.2)]
// CHECK:STDOUT:   %Self.2: %.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.type (%.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %a.patt => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc17_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc17_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(%a.loc17_15.2) {
// CHECK:STDOUT:   %a.loc17_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc17_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeat_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %const: type = const_type %C [template]
// CHECK:STDOUT:   %a: %const = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %const = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.generic: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = facet_type <@.1, @.1(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %const = value_param_pattern %a.patt.loc6_15.1, runtime_param<invalid> [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %const: type = const_type %C [template = constants.%const]
// CHECK:STDOUT:     %a.param: %const = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_15.1: %const = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type.1 = interface_decl @.1 [template = constants.%.generic] {
// CHECK:STDOUT:     %a.patt.loc17_15.1: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc17_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %const = value_param_pattern %a.patt.loc17_15.1, runtime_param<invalid> [symbolic = %a.patt.loc17_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %const.loc17_26: type = const_type %C [template = constants.%const]
// CHECK:STDOUT:     %const.loc17_19: type = const_type %const [template = constants.%const]
// CHECK:STDOUT:     %a.param: %const = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc17_15.1: %const = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc17_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_15.1: %const) {
// CHECK:STDOUT:   %a.loc6_15.2: %const = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_15.2: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(%a.loc17_15.1: %const) {
// CHECK:STDOUT:   %a.loc17_15.2: %const = bind_symbolic_name a, 0 [symbolic = %a.loc17_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc17_15.2: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc17_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.type: type = facet_type <@.1, @.1(%a.loc17_15.2)> [symbolic = %.type (constants.%.type.2)]
// CHECK:STDOUT:   %Self.2: %.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.type (%.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc17_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc17_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(%a.loc17_15.2) {
// CHECK:STDOUT:   %a.loc17_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc17_15.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
