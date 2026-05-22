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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/syntactic_merge.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/syntactic_merge.carbon

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

// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE+8]]:11: error: duplicate name `Foo` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface Foo(a:! C) {}
// CHECK:STDERR:           ^~~
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: two_file.carbon:7:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Foo(a:! C) {}
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE+8]]:11: error: duplicate name `Bar` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface Bar(a:! D) {}
// CHECK:STDERR:           ^~~
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE-14]]:1: in import [InImport]
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
// CHECK:STDERR: fail_alias_two_file.impl.carbon:[[@LINE+8]]:11: error: duplicate name `Foo` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface Foo(a:! D) {}
// CHECK:STDERR:           ^~~
// CHECK:STDERR: fail_alias_two_file.impl.carbon:[[@LINE-10]]:1: in import [InImport]
// CHECK:STDERR: alias_two_file.carbon:6:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Foo(a:! D) {}

// --- fail_repeat_const.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a:! const C);
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+11]]:19: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: interface Foo(a:! const (const C)) {}
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+7]]:25: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: interface Foo(a:! const (const C)) {}
// CHECK:STDERR:                         ^
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE-8]]:25: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: interface Foo(a:! const C);
// CHECK:STDERR:                         ^
// CHECK:STDERR:
interface Foo(a:! const (const C)) {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.43f: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.43f = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7fb: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self.412: %Foo.type.7fb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.44b: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.44b = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.205: type = facet_type <@Bar, @Bar(%a)> [symbolic]
// CHECK:STDOUT:   %Self.1f7: %Bar.type.205 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc7
// CHECK:STDOUT:     .Bar = %Bar.decl.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.ref [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.43f = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref.loc7 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type.43f = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %C.ref.loc8 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc8: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc8: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type.44b = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %D.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref.loc10: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc10_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type.44b = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %D.ref.loc11 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc11: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref.loc11: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc7_16.1)> [symbolic = %Foo.type (constants.%Foo.type.7fb)]
// CHECK:STDOUT:   %Self.loc8_22.2: @Foo.%Foo.type (%Foo.type.7fb) = symbolic_binding Self, 1 [symbolic = %Self.loc8_22.2 (constants.%Self.412)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_22.1: @Foo.%Foo.type (%Foo.type.7fb) = symbolic_binding Self, 1 [symbolic = %Self.loc8_22.2 (constants.%Self.412)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%a.loc10_16.2: %C) {
// CHECK:STDOUT:   %a.loc10_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%a.loc10_16.1)> [symbolic = %Bar.type (constants.%Bar.type.205)]
// CHECK:STDOUT:   %Self.loc11_22.2: @Bar.%Bar.type (%Bar.type.205) = symbolic_binding Self, 1 [symbolic = %Self.loc11_22.2 (constants.%Self.1f7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11_22.1: @Bar.%Bar.type (%Bar.type.205) = symbolic_binding Self, 1 [symbolic = %Self.loc11_22.2 (constants.%Self.1f7)]
// CHECK:STDOUT:     %Bar.WithSelf.decl = interface_with_self_decl @Bar [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self.412) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc10_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.WithSelf(constants.%a, constants.%Self.1f7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.6f0: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.6f0 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.771: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.771 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.6f0 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref.loc6 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc6: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_23.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_23.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.6f0 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref.loc7 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7: %C = symbolic_binding a, 0 [symbolic = %a.loc6_23.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_23.2: %C) {
// CHECK:STDOUT:   %a.loc6_23.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_23.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc6_23.1)> [symbolic = %Foo.type (constants.%Foo.type.771)]
// CHECK:STDOUT:   %Self.loc7_24.2: @Foo.%Foo.type (%Foo.type.771) = symbolic_binding Self, 1 [symbolic = %Self.loc7_24.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc7_24.1: @Foo.%Foo.type (%Foo.type.771) = symbolic_binding Self, 1 [symbolic = %Self.loc7_24.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_24.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_23.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.531e0f.1: type = generic_interface_type @Foo.loc6 [concrete]
// CHECK:STDOUT:   %Foo.generic.6d3a0f.1: %Foo.type.531e0f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.531e0f.2: type = generic_interface_type @Foo.loc14 [concrete]
// CHECK:STDOUT:   %Foo.generic.6d3a0f.2: %Foo.type.531e0f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.18d: type = facet_type <@Foo.loc14, @Foo.loc14(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.18d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.531e0f.1 = interface_decl @Foo.loc6 [concrete = constants.%Foo.generic.6d3a0f.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc14: %Foo.type.531e0f.2 = interface_decl @Foo.loc14 [concrete = constants.%Foo.generic.6d3a0f.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc14_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc14_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc6(%a.loc6_16.2: %C) {
// CHECK:STDOUT:   %a.loc6_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc14(%a.loc14_16.2: %C) {
// CHECK:STDOUT:   %a.loc14_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc14_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc14, @Foo.loc14(%a.loc14_16.1)> [symbolic = %Foo.type (constants.%Foo.type.18d)]
// CHECK:STDOUT:   %Self.loc14_24.2: @Foo.loc14.%Foo.type (%Foo.type.18d) = symbolic_binding Self, 1 [symbolic = %Self.loc14_24.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc14_24.1: @Foo.loc14.%Foo.type (%Foo.type.18d) = symbolic_binding Self, 1 [symbolic = %Self.loc14_24.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc14 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc14_24.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc6(constants.%a) {
// CHECK:STDOUT:   %a.loc6_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc14(constants.%a) {
// CHECK:STDOUT:   %a.loc14_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.09e: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.09e = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.01e: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.01e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.09e = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref.loc6 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc6: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.09e = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref.loc7 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_16.2: %C) {
// CHECK:STDOUT:   %a.loc6_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc6_16.1)> [symbolic = %Foo.type (constants.%Foo.type.01e)]
// CHECK:STDOUT:   %Self.loc7_24.2: @Foo.%Foo.type (%Foo.type.01e) = symbolic_binding Self, 1 [symbolic = %Self.loc7_24.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc7_24.1: @Foo.%Foo.type (%Foo.type.01e) = symbolic_binding Self, 1 [symbolic = %Self.loc7_24.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_24.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.ref [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc8_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc8_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%a.loc8_16.2: %C) {
// CHECK:STDOUT:   %a.loc8_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc8_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc8_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.b80382.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.d083e9.1: %Foo.type.b80382.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.b80382.2: type = generic_interface_type @Foo.loc12 [concrete]
// CHECK:STDOUT:   %Foo.generic.d083e9.2: %Foo.type.b80382.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bcb: type = facet_type <@Foo.loc12, @Foo.loc12(%a)> [symbolic]
// CHECK:STDOUT:   %Self.a96: %Foo.type.bcb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.0411d7.1: type = generic_interface_type @Bar.1 [concrete]
// CHECK:STDOUT:   %Bar.generic.7e8c7e.1: %Bar.type.0411d7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.0411d7.2: type = generic_interface_type @Bar.loc21 [concrete]
// CHECK:STDOUT:   %Bar.generic.7e8c7e.2: %Bar.type.0411d7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.2a5: type = facet_type <@Bar.loc21, @Bar.loc21(%a)> [symbolic]
// CHECK:STDOUT:   %Self.79f: %Bar.type.2a5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//two_file, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.Foo: %Foo.type.b80382.1 = import_ref Main//two_file, Foo, loaded [concrete = constants.%Foo.generic.d083e9.1]
// CHECK:STDOUT:   %Main.Bar: %Bar.type.0411d7.1 = import_ref Main//two_file, Bar, loaded [concrete = constants.%Bar.generic.7e8c7e.1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.6ef = import_ref Main//two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bb90b9.1: %C = import_ref Main//two_file, loc7_16, loaded [symbolic = @Foo.1.%a (constants.%a)]
// CHECK:STDOUT:   %Main.import_ref.bb90b9.2: %C = import_ref Main//two_file, loc8_16, loaded [symbolic = @Bar.1.%a (constants.%a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .Foo = imports.%Main.Foo
// CHECK:STDOUT:     .Bar = imports.%Main.Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_24.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_24.2 = import <none>
// CHECK:STDOUT:   %Foo.decl: %Foo.type.b80382.2 = interface_decl @Foo.loc12 [concrete = constants.%Foo.generic.d083e9.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc12_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc12_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.0411d7.2 = interface_decl @Bar.loc21 [concrete = constants.%Bar.generic.7e8c7e.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc21_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc21_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(imports.%Main.import_ref.bb90b9.1: %C) [from "two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc12(%a.loc12_16.2: %C) {
// CHECK:STDOUT:   %a.loc12_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc12_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc12, @Foo.loc12(%a.loc12_16.1)> [symbolic = %Foo.type (constants.%Foo.type.bcb)]
// CHECK:STDOUT:   %Self.loc12_22.2: @Foo.loc12.%Foo.type (%Foo.type.bcb) = symbolic_binding Self, 1 [symbolic = %Self.loc12_22.2 (constants.%Self.a96)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc12_22.1: @Foo.loc12.%Foo.type (%Foo.type.bcb) = symbolic_binding Self, 1 [symbolic = %Self.loc12_22.2 (constants.%Self.a96)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc12 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc12_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar.1(imports.%Main.import_ref.bb90b9.2: %C) [from "two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar.loc21(%a.loc21_16.2: %C) {
// CHECK:STDOUT:   %a.loc21_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc21_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar.loc21, @Bar.loc21(%a.loc21_16.1)> [symbolic = %Bar.type (constants.%Bar.type.2a5)]
// CHECK:STDOUT:   %Self.loc21_22.2: @Bar.loc21.%Bar.type (%Bar.type.2a5) = symbolic_binding Self, 1 [symbolic = %Self.loc21_22.2 (constants.%Self.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc21_22.1: @Bar.loc21.%Bar.type (%Bar.type.2a5) = symbolic_binding Self, 1 [symbolic = %Self.loc21_22.2 (constants.%Self.79f)]
// CHECK:STDOUT:     %Bar.WithSelf.decl = interface_with_self_decl @Bar.loc21 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc21_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "two_file.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc12(constants.%a) {
// CHECK:STDOUT:   %a.loc12_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self.a96) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.loc21(constants.%a) {
// CHECK:STDOUT:   %a.loc21_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.WithSelf(constants.%a, constants.%Self.79f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.18e74d.1: type = generic_interface_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.246c6b.1: %Foo.type.18e74d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %b: %C = symbolic_binding b, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.18e74d.2: type = generic_interface_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.generic.246c6b.2: %Foo.type.18e74d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.309: type = facet_type <@Foo.loc15, @Foo.loc15(%b)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.309 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.ref [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.18e74d.1 = interface_decl @Foo.loc7 [concrete = constants.%Foo.generic.246c6b.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.18e74d.2 = interface_decl @Foo.loc15 [concrete = constants.%Foo.generic.246c6b.2] {
// CHECK:STDOUT:     %b.patt: %pattern_type = symbolic_binding_pattern b, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc15_16.2: %C = symbolic_binding b, 0 [symbolic = %b.loc15_16.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc7(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc15(%b.loc15_16.2: %C) {
// CHECK:STDOUT:   %b.loc15_16.1: %C = symbolic_binding b, 0 [symbolic = %b.loc15_16.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc15, @Foo.loc15(%b.loc15_16.1)> [symbolic = %Foo.type (constants.%Foo.type.309)]
// CHECK:STDOUT:   %Self.loc15_22.2: @Foo.loc15.%Foo.type (%Foo.type.309) = symbolic_binding Self, 1 [symbolic = %Self.loc15_22.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_22.1: @Foo.loc15.%Foo.type (%Foo.type.309) = symbolic_binding Self, 1 [symbolic = %Self.loc15_22.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc15 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc7(constants.%a) {
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc15(constants.%b) {
// CHECK:STDOUT:   %b.loc15_16.1 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%b, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.4c0481.1: type = generic_interface_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.5b8364.1: %Foo.type.4c0481.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.4c0481.2: type = generic_interface_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.generic.5b8364.2: %Foo.type.4c0481.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.cbd: type = facet_type <@Foo.loc15, @Foo.loc15(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.cbd = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.ref [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.4c0481.1 = interface_decl @Foo.loc7 [concrete = constants.%Foo.generic.5b8364.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.4c0481.2 = interface_decl @Foo.loc15 [concrete = constants.%Foo.generic.5b8364.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc15_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc15_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc7(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc15(%a.loc15_16.2: %C) {
// CHECK:STDOUT:   %a.loc15_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc15_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc15, @Foo.loc15(%a.loc15_16.1)> [symbolic = %Foo.type (constants.%Foo.type.cbd)]
// CHECK:STDOUT:   %Self.loc15_22.2: @Foo.loc15.%Foo.type (%Foo.type.cbd) = symbolic_binding Self, 1 [symbolic = %Self.loc15_22.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_22.1: @Foo.loc15.%Foo.type (%Foo.type.cbd) = symbolic_binding Self, 1 [symbolic = %Self.loc15_22.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc15 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc7(constants.%a) {
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc15(constants.%a) {
// CHECK:STDOUT:   %a.loc15_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduced_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bab258.1: type = generic_interface_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.2ce101.1: %Foo.type.bab258.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bab258.2: type = generic_interface_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.generic.2ce101.2: %Foo.type.bab258.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.f80: type = facet_type <@Foo.loc15, @Foo.loc15(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.f80 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.ref [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.bab258.1 = interface_decl @Foo.loc7 [concrete = constants.%Foo.generic.2ce101.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.bab258.2 = interface_decl @Foo.loc15 [concrete = constants.%Foo.generic.2ce101.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc15_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc15_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc7(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc15(%a.loc15_16.2: %C) {
// CHECK:STDOUT:   %a.loc15_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc15_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc15, @Foo.loc15(%a.loc15_16.1)> [symbolic = %Foo.type (constants.%Foo.type.f80)]
// CHECK:STDOUT:   %Self.loc15_24.2: @Foo.loc15.%Foo.type (%Foo.type.f80) = symbolic_binding Self, 1 [symbolic = %Self.loc15_24.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_24.1: @Foo.loc15.%Foo.type (%Foo.type.f80) = symbolic_binding Self, 1 [symbolic = %Self.loc15_24.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc15 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_24.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc7(constants.%a) {
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc15(constants.%a) {
// CHECK:STDOUT:   %a.loc15_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_16.2: %C) {
// CHECK:STDOUT:   %a.loc6_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.48e6f6.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.c0a75a.1: %Foo.type.48e6f6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.48e6f6.2: type = generic_interface_type @Foo.loc17 [concrete]
// CHECK:STDOUT:   %Foo.generic.c0a75a.2: %Foo.type.48e6f6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.0ed: type = facet_type <@Foo.loc17, @Foo.loc17(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.0ed = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//alias_two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.Foo: %Foo.type.48e6f6.1 = import_ref Main//alias_two_file, Foo, loaded [concrete = constants.%Foo.generic.c0a75a.1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//alias_two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.3ad = import_ref Main//alias_two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.6f0: %C = import_ref Main//alias_two_file, loc6_16, loaded [symbolic = @Foo.1.%a (constants.%a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Foo = imports.%Main.Foo
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_30.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_30.2 = import <none>
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.ref [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type.48e6f6.2 = interface_decl @Foo.loc17 [concrete = constants.%Foo.generic.c0a75a.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc17_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc17_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(imports.%Main.import_ref.6f0: %C) [from "alias_two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc17(%a.loc17_16.2: %C) {
// CHECK:STDOUT:   %a.loc17_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc17_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc17, @Foo.loc17(%a.loc17_16.1)> [symbolic = %Foo.type (constants.%Foo.type.0ed)]
// CHECK:STDOUT:   %Self.loc17_22.2: @Foo.loc17.%Foo.type (%Foo.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc17_22.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc17_22.1: @Foo.loc17.%Foo.type (%Foo.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc17_22.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc17 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc17_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "alias_two_file.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.3ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc17(constants.%a) {
// CHECK:STDOUT:   %a.loc17_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeat_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %a: %const = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.b6229b.1: type = generic_interface_type @Foo.loc6 [concrete]
// CHECK:STDOUT:   %Foo.generic.7b7db8.1: %Foo.type.b6229b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.b6229b.2: type = generic_interface_type @Foo.loc18 [concrete]
// CHECK:STDOUT:   %Foo.generic.7b7db8.2: %Foo.type.b6229b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5ba: type = facet_type <@Foo.loc18, @Foo.loc18(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.5ba = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.b6229b.1 = interface_decl @Foo.loc6 [concrete = constants.%Foo.generic.7b7db8.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_16.2: %const = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc18: %Foo.type.b6229b.2 = interface_decl @Foo.loc18 [concrete = constants.%Foo.generic.7b7db8.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %const.loc18_19 [concrete = constants.%const] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc18_26: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:       %const.loc18_19: type = const_type %const.loc18_26 [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc18_16.2: %const = symbolic_binding a, 0 [symbolic = %a.loc18_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc6(%a.loc6_16.2: %const) {
// CHECK:STDOUT:   %a.loc6_16.1: %const = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc18(%a.loc18_16.2: %const) {
// CHECK:STDOUT:   %a.loc18_16.1: %const = symbolic_binding a, 0 [symbolic = %a.loc18_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc18, @Foo.loc18(%a.loc18_16.1)> [symbolic = %Foo.type (constants.%Foo.type.5ba)]
// CHECK:STDOUT:   %Self.loc18_36.2: @Foo.loc18.%Foo.type (%Foo.type.5ba) = symbolic_binding Self, 1 [symbolic = %Self.loc18_36.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc18_36.1: @Foo.loc18.%Foo.type (%Foo.type.5ba) = symbolic_binding Self, 1 [symbolic = %Self.loc18_36.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc18 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc18_36.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc6(constants.%a) {
// CHECK:STDOUT:   %a.loc6_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc18(constants.%a) {
// CHECK:STDOUT:   %a.loc18_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
