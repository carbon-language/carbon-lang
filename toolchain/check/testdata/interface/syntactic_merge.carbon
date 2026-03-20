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
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-4]]:15: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR:               ^
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
// CHECK:STDOUT:   %Foo.type.5ed: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.5ed = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self.f8c: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.d8f: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.d8f = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.c4b: type = facet_type <@Bar, @Bar(%a)> [symbolic]
// CHECK:STDOUT:   %Self.0fc: %Bar.type.c4b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc20
// CHECK:STDOUT:     .Bar = %Bar.decl.loc37
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc20: %Foo.type.5ed = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc28: %Foo.type.5ed = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26: type = splice_block %C.ref.loc26 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc24: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc37: %Bar.type.d8f = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc35: type = splice_block %D.ref.loc35 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref.loc35: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc33.2: %C = symbolic_binding a, 0 [symbolic = %a.loc33.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc45: %Bar.type.d8f = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc43: type = splice_block %D.ref.loc43 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref.loc43: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc41: %C = symbolic_binding a, 0 [symbolic = %a.loc33.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc16.2: %C) {
// CHECK:STDOUT:   %a.loc16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc16.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc28.2: @Foo.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc28.2 (constants.%Self.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc28.1: @Foo.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc28.2 (constants.%Self.f8c)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc28.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%a.loc33.2: %C) {
// CHECK:STDOUT:   %a.loc33.1: %C = symbolic_binding a, 0 [symbolic = %a.loc33.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%a.loc33.1)> [symbolic = %Bar.type (constants.%Bar.type.c4b)]
// CHECK:STDOUT:   %Self.loc45.2: @Bar.%Bar.type (%Bar.type.c4b) = symbolic_binding Self, 1 [symbolic = %Self.loc45.2 (constants.%Self.0fc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc45.1: @Bar.%Bar.type (%Bar.type.c4b) = symbolic_binding Self, 1 [symbolic = %Self.loc45.2 (constants.%Self.0fc)]
// CHECK:STDOUT:     %Bar.WithSelf.decl = interface_with_self_decl @Bar [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc45.1
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
// CHECK:STDOUT:   %a.loc16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self.f8c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc33.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.WithSelf(constants.%a, constants.%Self.0fc) {}
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
// CHECK:STDOUT:   %Foo.type.5ed: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.5ed = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc17: %Foo.type.5ed = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %C.ref.loc15 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc13.2: %C = symbolic_binding a, 0 [symbolic = %a.loc13.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc27: %Foo.type.5ed = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25: type = splice_block %C.ref.loc25 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc25: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc23: %C = symbolic_binding a, 0 [symbolic = %a.loc13.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc13.2: %C) {
// CHECK:STDOUT:   %a.loc13.1: %C = symbolic_binding a, 0 [symbolic = %a.loc13.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc13.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc27.2: @Foo.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc27.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc27.1: @Foo.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc27.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc27.1
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
// CHECK:STDOUT:   %a.loc13.1 => constants.%a
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
// CHECK:STDOUT:   %Foo.type.5ede67.1: type = generic_interface_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.1: %Foo.type.5ede67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5ede67.2: type = generic_interface_type @Foo.loc25 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.2: %Foo.type.5ede67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo.loc25, @Foo.loc25(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.5ede67.1 = interface_decl @Foo.loc15 [concrete = constants.%Foo.generic.fda91f.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11.2: %C = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc25: %Foo.type.5ede67.2 = interface_decl @Foo.loc25 [concrete = constants.%Foo.generic.fda91f.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc19.2: %C = symbolic_binding a, 0 [symbolic = %a.loc19.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc15(%a.loc11.2: %C) {
// CHECK:STDOUT:   %a.loc11.1: %C = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc25(%a.loc19.2: %C) {
// CHECK:STDOUT:   %a.loc19.1: %C = symbolic_binding a, 0 [symbolic = %a.loc19.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc25, @Foo.loc25(%a.loc19.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc25.2: @Foo.loc25.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc25.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc25.1: @Foo.loc25.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc25.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc25 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc25.1
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
// CHECK:STDOUT: specific @Foo.loc15(constants.%a) {
// CHECK:STDOUT:   %a.loc11.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc25(constants.%a) {
// CHECK:STDOUT:   %a.loc19.1 => constants.%a
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
// CHECK:STDOUT:   %Foo.type.5ed: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.5ed = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.5ed = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %C.ref.loc13 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11.2: %C = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc23: %Foo.type.5ed = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21: type = splice_block %C.ref.loc21 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc19: %C = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc11.2: %C) {
// CHECK:STDOUT:   %a.loc11.1: %C = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc11.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc23.2: @Foo.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc23.1: @Foo.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc23.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc23.1
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
// CHECK:STDOUT:   %a.loc11.1 => constants.%a
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
// CHECK:STDOUT:   %D: type = alias_binding D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc24.2: %C = symbolic_binding a, 0 [symbolic = %a.loc24.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc16.2: %C) {
// CHECK:STDOUT:   %a.loc16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%a.loc24.2: %C) {
// CHECK:STDOUT:   %a.loc24.1: %C = symbolic_binding a, 0 [symbolic = %a.loc24.1 (constants.%a)]
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
// CHECK:STDOUT:   %a.loc16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc24.1 => constants.%a
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
// CHECK:STDOUT:   %Foo.type.5ede67.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.1: %Foo.type.5ede67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5ede67.2: type = generic_interface_type @Foo.loc12 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.2: %Foo.type.5ede67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo.loc12, @Foo.loc12(%a)> [symbolic]
// CHECK:STDOUT:   %Self.f8c: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.d8fa75.1: type = generic_interface_type @Bar.1 [concrete]
// CHECK:STDOUT:   %Bar.generic.9a48ff.1: %Bar.type.d8fa75.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.d8fa75.2: type = generic_interface_type @Bar.loc21 [concrete]
// CHECK:STDOUT:   %Bar.generic.9a48ff.2: %Bar.type.d8fa75.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.c4b: type = facet_type <@Bar.loc21, @Bar.loc21(%a)> [symbolic]
// CHECK:STDOUT:   %Self.0fc: %Bar.type.c4b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//two_file, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.Foo: %Foo.type.5ede67.1 = import_ref Main//two_file, Foo, loaded [concrete = constants.%Foo.generic.fda91f.1]
// CHECK:STDOUT:   %Main.Bar: %Bar.type.d8fa75.1 = import_ref Main//two_file, Bar, loaded [concrete = constants.%Bar.generic.9a48ff.1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//two_file, loc7, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a7fd84.1: %C = import_ref Main//two_file, loc16, loaded [symbolic = @Foo.1.%a (constants.%a)]
// CHECK:STDOUT:   %Main.import_ref.a7fd84.2: %C = import_ref Main//two_file, loc24, loaded [symbolic = @Bar.1.%a (constants.%a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .Foo = imports.%Main.Foo
// CHECK:STDOUT:     .Bar = imports.%Main.Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   %Foo.decl: %Foo.type.5ede67.2 = interface_decl @Foo.loc12 [concrete = constants.%Foo.generic.fda91f.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc8.2: %C = symbolic_binding a, 0 [symbolic = %a.loc8.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.d8fa75.2 = interface_decl @Bar.loc21 [concrete = constants.%Bar.generic.9a48ff.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc17.2: %C = symbolic_binding a, 0 [symbolic = %a.loc17.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(imports.%Main.import_ref.a7fd84.1: %C) [from "two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc12(%a.loc8.2: %C) {
// CHECK:STDOUT:   %a.loc8.1: %C = symbolic_binding a, 0 [symbolic = %a.loc8.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc12, @Foo.loc12(%a.loc8.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc12.2: @Foo.loc12.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc12.2 (constants.%Self.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc12.1: @Foo.loc12.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc12.2 (constants.%Self.f8c)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc12 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc12.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar.1(imports.%Main.import_ref.a7fd84.2: %C) [from "two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar.loc21(%a.loc17.2: %C) {
// CHECK:STDOUT:   %a.loc17.1: %C = symbolic_binding a, 0 [symbolic = %a.loc17.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar.loc21, @Bar.loc21(%a.loc17.1)> [symbolic = %Bar.type (constants.%Bar.type.c4b)]
// CHECK:STDOUT:   %Self.loc21.2: @Bar.loc21.%Bar.type (%Bar.type.c4b) = symbolic_binding Self, 1 [symbolic = %Self.loc21.2 (constants.%Self.0fc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc21.1: @Bar.loc21.%Bar.type (%Bar.type.c4b) = symbolic_binding Self, 1 [symbolic = %Self.loc21.2 (constants.%Self.0fc)]
// CHECK:STDOUT:     %Bar.WithSelf.decl = interface_with_self_decl @Bar.loc21 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc21.1
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc12(constants.%a) {
// CHECK:STDOUT:   %a.loc8.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self.f8c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.loc21(constants.%a) {
// CHECK:STDOUT:   %a.loc17.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.WithSelf(constants.%a, constants.%Self.0fc) {}
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
// CHECK:STDOUT:   %Foo.type.5ede67.1: type = generic_interface_type @Foo.loc20 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.1: %Foo.type.5ede67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %b: %C = symbolic_binding b, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5ede67.2: type = generic_interface_type @Foo.loc28 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.2: %Foo.type.5ede67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo.loc28, @Foo.loc28(%b)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc20: %Foo.type.5ede67.1 = interface_decl @Foo.loc20 [concrete = constants.%Foo.generic.fda91f.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc28: %Foo.type.5ede67.2 = interface_decl @Foo.loc28 [concrete = constants.%Foo.generic.fda91f.2] {
// CHECK:STDOUT:     %b.patt: %pattern_type = symbolic_binding_pattern b, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc24.2: %C = symbolic_binding b, 0 [symbolic = %b.loc24.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc20(%a.loc16.2: %C) {
// CHECK:STDOUT:   %a.loc16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc28(%b.loc24.2: %C) {
// CHECK:STDOUT:   %b.loc24.1: %C = symbolic_binding b, 0 [symbolic = %b.loc24.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc28, @Foo.loc28(%b.loc24.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc28.2: @Foo.loc28.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc28.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc28.1: @Foo.loc28.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc28.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc28 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc28.1
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
// CHECK:STDOUT: specific @Foo.loc20(constants.%a) {
// CHECK:STDOUT:   %a.loc16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc28(constants.%b) {
// CHECK:STDOUT:   %b.loc24.1 => constants.%b
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
// CHECK:STDOUT:   %Foo.type.5ede67.1: type = generic_interface_type @Foo.loc20 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.1: %Foo.type.5ede67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5ede67.2: type = generic_interface_type @Foo.loc28 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.2: %Foo.type.5ede67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo.loc28, @Foo.loc28(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc20: %Foo.type.5ede67.1 = interface_decl @Foo.loc20 [concrete = constants.%Foo.generic.fda91f.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc28: %Foo.type.5ede67.2 = interface_decl @Foo.loc28 [concrete = constants.%Foo.generic.fda91f.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc24.2: %C = symbolic_binding a, 0 [symbolic = %a.loc24.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc20(%a.loc16.2: %C) {
// CHECK:STDOUT:   %a.loc16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc28(%a.loc24.2: %C) {
// CHECK:STDOUT:   %a.loc24.1: %C = symbolic_binding a, 0 [symbolic = %a.loc24.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc28, @Foo.loc28(%a.loc24.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc28.2: @Foo.loc28.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc28.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc28.1: @Foo.loc28.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc28.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc28 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc28.1
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
// CHECK:STDOUT: specific @Foo.loc20(constants.%a) {
// CHECK:STDOUT:   %a.loc16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc28(constants.%a) {
// CHECK:STDOUT:   %a.loc24.1 => constants.%a
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
// CHECK:STDOUT:   %Foo.type.5ede67.1: type = generic_interface_type @Foo.loc22 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.1: %Foo.type.5ede67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5ede67.2: type = generic_interface_type @Foo.loc32 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.2: %Foo.type.5ede67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo.loc32, @Foo.loc32(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc22: %Foo.type.5ede67.1 = interface_decl @Foo.loc22 [concrete = constants.%Foo.generic.fda91f.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc32: %Foo.type.5ede67.2 = interface_decl @Foo.loc32 [concrete = constants.%Foo.generic.fda91f.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc26.2: %C = symbolic_binding a, 0 [symbolic = %a.loc26.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc22(%a.loc16.2: %C) {
// CHECK:STDOUT:   %a.loc16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc32(%a.loc26.2: %C) {
// CHECK:STDOUT:   %a.loc26.1: %C = symbolic_binding a, 0 [symbolic = %a.loc26.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc32, @Foo.loc32(%a.loc26.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc32.2: @Foo.loc32.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc32.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc32.1: @Foo.loc32.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc32.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc32 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc32.1
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
// CHECK:STDOUT: specific @Foo.loc22(constants.%a) {
// CHECK:STDOUT:   %a.loc16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc32(constants.%a) {
// CHECK:STDOUT:   %a.loc26.1 => constants.%a
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
// CHECK:STDOUT:     %.loc13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11.2: %C = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc11.2: %C) {
// CHECK:STDOUT:   %a.loc11.1: %C = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
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
// CHECK:STDOUT:   %a.loc11.1 => constants.%a
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
// CHECK:STDOUT:   %Foo.type.5ede67.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.1: %Foo.type.5ede67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5ede67.2: type = generic_interface_type @Foo.loc17 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.2: %Foo.type.5ede67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.6f9: type = facet_type <@Foo.loc17, @Foo.loc17(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.6f9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//alias_two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.Foo: %Foo.type.5ede67.1 = import_ref Main//alias_two_file, Foo, loaded [concrete = constants.%Foo.generic.fda91f.1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//alias_two_file, loc7, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//alias_two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a7f: %C = import_ref Main//alias_two_file, loc11, loaded [symbolic = @Foo.1.%a (constants.%a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Foo = imports.%Main.Foo
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type.5ede67.2 = interface_decl @Foo.loc17 [concrete = constants.%Foo.generic.fda91f.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc13.2: %C = symbolic_binding a, 0 [symbolic = %a.loc13.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(imports.%Main.import_ref.a7f: %C) [from "alias_two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc17(%a.loc13.2: %C) {
// CHECK:STDOUT:   %a.loc13.1: %C = symbolic_binding a, 0 [symbolic = %a.loc13.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc17, @Foo.loc17(%a.loc13.1)> [symbolic = %Foo.type (constants.%Foo.type.6f9)]
// CHECK:STDOUT:   %Self.loc17.2: @Foo.loc17.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc17.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc17.1: @Foo.loc17.%Foo.type (%Foo.type.6f9) = symbolic_binding Self, 1 [symbolic = %Self.loc17.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc17 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc17.1
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc17(constants.%a) {
// CHECK:STDOUT:   %a.loc13.1 => constants.%a
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
// CHECK:STDOUT:   %Foo.type.5ede67.1: type = generic_interface_type @Foo.loc16 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.1: %Foo.type.5ede67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5ede67.2: type = generic_interface_type @Foo.loc28 [concrete]
// CHECK:STDOUT:   %Foo.generic.fda91f.2: %Foo.type.5ede67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.442: type = facet_type <@Foo.loc28, @Foo.loc28(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.442 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc16: %Foo.type.5ede67.1 = interface_decl @Foo.loc16 [concrete = constants.%Foo.generic.fda91f.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11.2: %const = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc28: %Foo.type.5ede67.2 = interface_decl @Foo.loc28 [concrete = constants.%Foo.generic.fda91f.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22: type = splice_block %const.loc22 [concrete = constants.%const] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc24: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:       %const.loc22: type = const_type %const.loc24 [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc20.2: %const = symbolic_binding a, 0 [symbolic = %a.loc20.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc16(%a.loc11.2: %const) {
// CHECK:STDOUT:   %a.loc11.1: %const = symbolic_binding a, 0 [symbolic = %a.loc11.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc28(%a.loc20.2: %const) {
// CHECK:STDOUT:   %a.loc20.1: %const = symbolic_binding a, 0 [symbolic = %a.loc20.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc28, @Foo.loc28(%a.loc20.1)> [symbolic = %Foo.type (constants.%Foo.type.442)]
// CHECK:STDOUT:   %Self.loc28.2: @Foo.loc28.%Foo.type (%Foo.type.442) = symbolic_binding Self, 1 [symbolic = %Self.loc28.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc28.1: @Foo.loc28.%Foo.type (%Foo.type.442) = symbolic_binding Self, 1 [symbolic = %Self.loc28.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc28 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc28.1
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
// CHECK:STDOUT: specific @Foo.loc16(constants.%a) {
// CHECK:STDOUT:   %a.loc11.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc28(constants.%a) {
// CHECK:STDOUT:   %a.loc20.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
