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
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.538: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self.a71: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.982: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.982 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.6a9: type = facet_type <@Bar, @Bar(%a)> [symbolic]
// CHECK:STDOUT:   %Self.cec: %Bar.type.6a9 = bind_symbolic_name Self, 1 [symbolic]
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
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.538 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type.538 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc8: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type.982 = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc10_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc10_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc10_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type.982 = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc10_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc11: %C = bind_symbolic_name a, 0 [symbolic = %a.loc10_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc7_15.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a71)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a71)]
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
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%a.loc10_15.2)> [symbolic = %Bar.type (constants.%Bar.type.6a9)]
// CHECK:STDOUT:   %Self.2: @Bar.%Bar.type (%Bar.type.6a9) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.cec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Bar.%Bar.type (%Bar.type.6a9) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.cec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(%a.loc7_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc10_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc10_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(%a.loc10_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.538: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.538 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_21.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_21.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6_21.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_21.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.538 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_21.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_21.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_21.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_21.1: %C) {
// CHECK:STDOUT:   %a.loc6_21.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_21.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_21.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_21.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc6_21.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_21.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_21.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(%a.loc6_21.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5380b8.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.1: %Foo.type.5380b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5380b8.2: type = generic_interface_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.2: %Foo.type.5380b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo.2, @Foo.2(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.5380b8.1 = interface_decl @Foo.1 [concrete = constants.%Foo.generic.ec3175.1] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc14: %Foo.type.5380b8.2 = interface_decl @Foo.2 [concrete = constants.%Foo.generic.ec3175.2] {
// CHECK:STDOUT:     %a.patt.loc14_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc14_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc14_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(%a.loc6_15.1: %C) {
// CHECK:STDOUT:   %a.loc6_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.2(%a.loc14_15.1: %C) {
// CHECK:STDOUT:   %a.loc14_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc14_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc14_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.2, @Foo.2(%a.loc14_15.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc14_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc14_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(%a.loc14_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.538: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.538 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.538 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.538 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_15.1: %C) {
// CHECK:STDOUT:   %a.loc6_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc6_15.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(%a.loc6_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc8_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc8_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc8_15.2 (constants.%a)]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc8_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc8_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5380b8.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.1: %Foo.type.5380b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5380b8.2: type = generic_interface_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.2: %Foo.type.5380b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo.2, @Foo.2(%a)> [symbolic]
// CHECK:STDOUT:   %Self.a71: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.982aac.1: type = generic_interface_type @Bar.1 [concrete]
// CHECK:STDOUT:   %Bar.generic.4bda5e.1: %Bar.type.982aac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.982aac.2: type = generic_interface_type @Bar.2 [concrete]
// CHECK:STDOUT:   %Bar.generic.4bda5e.2: %Bar.type.982aac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.6a9: type = facet_type <@Bar.2, @Bar.2(%a)> [symbolic]
// CHECK:STDOUT:   %Self.cec: %Bar.type.6a9 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//two_file, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.Foo: %Foo.type.5380b8.1 = import_ref Main//two_file, Foo, loaded [concrete = constants.%Foo.generic.ec3175.1]
// CHECK:STDOUT:   %Main.Bar: %Bar.type.982aac.1 = import_ref Main//two_file, Bar, loaded [concrete = constants.%Bar.generic.4bda5e.1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//two_file, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f97b44.1: %C = import_ref Main//two_file, loc7_15, loaded [symbolic = @Foo.1.%a (constants.%a)]
// CHECK:STDOUT:   %Main.import_ref.f97b44.2: %C = import_ref Main//two_file, loc8_15, loaded [symbolic = @Bar.1.%a (constants.%a)]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type.5380b8.2 = interface_decl @Foo.2 [concrete = constants.%Foo.generic.ec3175.2] {
// CHECK:STDOUT:     %a.patt.loc12_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc12_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc12_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc12_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.982aac.2 = interface_decl @Bar.2 [concrete = constants.%Bar.generic.4bda5e.2] {
// CHECK:STDOUT:     %a.patt.loc21_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc21_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc21_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc21_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(imports.%Main.import_ref.f97b44.1: %C) [from "two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.2(%a.loc12_15.1: %C) {
// CHECK:STDOUT:   %a.loc12_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc12_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc12_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc12_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.2, @Foo.2(%a.loc12_15.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a71)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a71)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar.1(imports.%Main.import_ref.f97b44.2: %C) [from "two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar.2(%a.loc21_15.1: %C) {
// CHECK:STDOUT:   %a.loc21_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc21_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc21_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc21_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar.2, @Bar.2(%a.loc21_15.2)> [symbolic = %Bar.type (constants.%Bar.type.6a9)]
// CHECK:STDOUT:   %Self.2: @Bar.2.%Bar.type (%Bar.type.6a9) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.cec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Bar.2.%Bar.type (%Bar.type.6a9) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.cec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "two_file.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %a.patt => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc12_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc12_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(%a.loc12_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %a.patt => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.2(constants.%a) {
// CHECK:STDOUT:   %a.loc21_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc21_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.2(%a.loc21_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5380b8.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.1: %Foo.type.5380b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %b: %C = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5380b8.2: type = generic_interface_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.2: %Foo.type.5380b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo.2, @Foo.2(%b)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
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
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.5380b8.1 = interface_decl @Foo.1 [concrete = constants.%Foo.generic.ec3175.1] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.5380b8.2 = interface_decl @Foo.2 [concrete = constants.%Foo.generic.ec3175.2] {
// CHECK:STDOUT:     %b.patt.loc15_15.1: %C = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_15.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %b.loc15_15.1: %C = bind_symbolic_name b, 0 [symbolic = %b.loc15_15.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.2(%b.loc15_15.1: %C) {
// CHECK:STDOUT:   %b.loc15_15.2: %C = bind_symbolic_name b, 0 [symbolic = %b.loc15_15.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc15_15.2: %C = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_15.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.2, @Foo.2(%b.loc15_15.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%b) {
// CHECK:STDOUT:   %b.loc15_15.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc15_15.2 => constants.%b.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(%b.loc15_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5380b8.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.1: %Foo.type.5380b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5380b8.2: type = generic_interface_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.2: %Foo.type.5380b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo.2, @Foo.2(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
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
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.5380b8.1 = interface_decl @Foo.1 [concrete = constants.%Foo.generic.ec3175.1] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.5380b8.2 = interface_decl @Foo.2 [concrete = constants.%Foo.generic.ec3175.2] {
// CHECK:STDOUT:     %a.patt.loc15_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc15_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.2(%a.loc15_15.1: %C) {
// CHECK:STDOUT:   %a.loc15_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc15_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.2, @Foo.2(%a.loc15_15.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc15_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(%a.loc15_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduced_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5380b8.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.1: %Foo.type.5380b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5380b8.2: type = generic_interface_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.2: %Foo.type.5380b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo.2, @Foo.2(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
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
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.5380b8.1 = interface_decl @Foo.1 [concrete = constants.%Foo.generic.ec3175.1] {
// CHECK:STDOUT:     %a.patt.loc7_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.5380b8.2 = interface_decl @Foo.2 [concrete = constants.%Foo.generic.ec3175.2] {
// CHECK:STDOUT:     %a.patt.loc15_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc15_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(%a.loc7_15.1: %C) {
// CHECK:STDOUT:   %a.loc7_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.2(%a.loc15_15.1: %C) {
// CHECK:STDOUT:   %a.loc15_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc15_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.2, @Foo.2(%a.loc15_15.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc7_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc15_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(%a.loc15_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:     %a.patt.loc6_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5380b8.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.1: %Foo.type.5380b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5380b8.2: type = generic_interface_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.2: %Foo.type.5380b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.7d0: type = facet_type <@Foo.2, @Foo.2(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.7d0 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//alias_two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.Foo: %Foo.type.5380b8.1 = import_ref Main//alias_two_file, Foo, loaded [concrete = constants.%Foo.generic.ec3175.1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//alias_two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//alias_two_file, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f97: %C = import_ref Main//alias_two_file, loc6_15, loaded [symbolic = @Foo.1.%a (constants.%a)]
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
// CHECK:STDOUT:   %D: type = bind_alias D, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type.5380b8.2 = interface_decl @Foo.2 [concrete = constants.%Foo.generic.ec3175.2] {
// CHECK:STDOUT:     %a.patt.loc17_15.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc17_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc17_15.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc17_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(imports.%Main.import_ref.f97: %C) [from "alias_two_file.carbon"] {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.2(%a.loc17_15.1: %C) {
// CHECK:STDOUT:   %a.loc17_15.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc17_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc17_15.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc17_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.2, @Foo.2(%a.loc17_15.2)> [symbolic = %Foo.type (constants.%Foo.type.7d0)]
// CHECK:STDOUT:   %Self.2: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.2.%Foo.type (%Foo.type.7d0) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "alias_two_file.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT:   %a.patt => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc17_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc17_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(%a.loc17_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeat_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %a: %const = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %const = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.5380b8.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.1: %Foo.type.5380b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.5380b8.2: type = generic_interface_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.ec3175.2: %Foo.type.5380b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.924: type = facet_type <@Foo.2, @Foo.2(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.924 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.5380b8.1 = interface_decl @Foo.1 [concrete = constants.%Foo.generic.ec3175.1] {
// CHECK:STDOUT:     %a.patt.loc6_15.1: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_15.1: %const = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc18: %Foo.type.5380b8.2 = interface_decl @Foo.2 [concrete = constants.%Foo.generic.ec3175.2] {
// CHECK:STDOUT:     %a.patt.loc18_15.1: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc18_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %const.loc18_19 [concrete = constants.%const] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc18_26: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:       %const.loc18_19: type = const_type %const.loc18_26 [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc18_15.1: %const = bind_symbolic_name a, 0 [symbolic = %a.loc18_15.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(%a.loc6_15.1: %const) {
// CHECK:STDOUT:   %a.loc6_15.2: %const = bind_symbolic_name a, 0 [symbolic = %a.loc6_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_15.2: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.2(%a.loc18_15.1: %const) {
// CHECK:STDOUT:   %a.loc18_15.2: %const = bind_symbolic_name a, 0 [symbolic = %a.loc18_15.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc18_15.2: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc18_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.2, @Foo.2(%a.loc18_15.2)> [symbolic = %Foo.type (constants.%Foo.type.924)]
// CHECK:STDOUT:   %Self.2: @Foo.2.%Foo.type (%Foo.type.924) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.2.%Foo.type (%Foo.type.924) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc6_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc18_15.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc18_15.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(%a.loc18_15.2) {}
// CHECK:STDOUT:
