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

interface Foo(a: C);
interface Foo(a: C) {}

interface Bar(a: D);
interface Bar(a: D) {}

// --- spacing.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo [ ] ( a : C );
interface Foo[](a: C) {}

// --- fail_parens.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a: C);
// CHECK:STDERR: fail_parens.carbon:[[@LINE+3]]:18: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fail_parens.carbon:[[@LINE-2]]:18: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:
interface Foo(a: (C)) {}

// --- fail_raw_identifier.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a: C);
// CHECK:STDERR: fail_raw_identifier.carbon:[[@LINE+3]]:18-20: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fail_raw_identifier.carbon:[[@LINE-2]]:18: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:
interface Foo(a: r#C) {}

// --- two_file.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a: C);
interface Bar(a: D);

// --- fail_todo_two_file.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE+4]]:11-13: error: duplicate name `Foo` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_todo_two_file.impl.carbon:[[@LINE-3]]:1 [InImport]
// CHECK:STDERR: two_file.carbon:7:1-20: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
interface Foo(a: C) {}
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE+4]]:11-13: error: duplicate name `Bar` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_todo_two_file.impl.carbon:[[@LINE-8]]:1 [InImport]
// CHECK:STDERR: two_file.carbon:8:1-20: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
interface Bar(a: D) {}

// --- fail_name_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a: C);
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE+3]]:15-18: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-2]]:15-18: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:
interface Foo(b: D) {}

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a: C);
// CHECK:STDERR: fail_alias.carbon:[[@LINE+3]]:18: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fail_alias.carbon:[[@LINE-2]]:18: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:
interface Foo(a: D) {}

// --- fail_deduced_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo[a: C]();
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE+3]]:18: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE-2]]:18: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:
interface Foo[a: D]() {}

// --- alias_two_file.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a: C);

// --- fail_alias_two_file.impl.carbon

impl library "[[@TEST_NAME]]";

alias D = C;

// TODO: This fails because importing interfaces doesn't work well. It should
// fail due to `C` versus `D`, but may succeed if importing interfaces is fixed
// before syntax matching on imports is supported.
// CHECK:STDERR: fail_alias_two_file.impl.carbon:[[@LINE+4]]:11-13: error: duplicate name `Foo` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: imported from: fail_alias_two_file.impl.carbon:[[@LINE-8]]:1 [InImport]
// CHECK:STDERR: alias_two_file.carbon:6:1-20: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR:
interface Foo(a: D) {}

// --- fail_repeat_const.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a: const C);
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+5]]:18-32: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR:
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+3]]:24: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE-4]]:24: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:
interface Foo(a: const (const C)) {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.bc6 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self.a95: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.3f0: type = generic_interface_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type.3f0 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.ab7: type = facet_type <@Bar, @Bar(%a)> [symbolic]
// CHECK:STDOUT:   %Self.a90: %Bar.type.ab7 = symbolic_binding Self, 1 [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.bc6 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc8: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref.loc7 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen.loc7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type.bc6 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc8: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %C.ref.loc8 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen.loc8: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc8: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type.3f0 = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc11: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %D.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen.loc10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref.loc10: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc10_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type.3f0 = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc11: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %D.ref.loc11 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen.loc11: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref.loc11: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc7_16.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc8_21.2: @Foo.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc8_21.2 (constants.%Self.a95)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_21.1: @Foo.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc8_21.2 (constants.%Self.a95)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_21.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%a.loc10_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc10: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc10_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar, @Bar(%a.loc10_16.1)> [symbolic = %Bar.type (constants.%Bar.type.ab7)]
// CHECK:STDOUT:   %Self.loc11_21.2: @Bar.%Bar.type (%Bar.type.ab7) = symbolic_binding Self, 1 [symbolic = %Self.loc11_21.2 (constants.%Self.a90)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11_21.1: @Bar.%Bar.type (%Bar.type.ab7) = symbolic_binding Self, 1 [symbolic = %Self.loc11_21.2 (constants.%Self.a90)]
// CHECK:STDOUT:     %Bar.WithSelf.decl = interface_with_self_decl @Bar [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11_21.1
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
// CHECK:STDOUT:   %a.patt.loc7 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self.a95) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc10 => constants.%a.patt
// CHECK:STDOUT:   %a.loc10_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.WithSelf(constants.%a, constants.%Self.a90) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6: type = generic_interface_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type.bc6 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo, @Foo(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.bc6 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref.loc6 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen.loc6: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref.loc6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_23.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_23.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.bc6 = interface_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref.loc7 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen.loc7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7: %C = symbolic_binding a, 0 [symbolic = %a.loc6_23.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_23.2: %C) {
// CHECK:STDOUT:   %a.patt.loc6: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_23.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_23.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo, @Foo(%a.loc6_23.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc7_23.2: @Foo.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc7_23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc7_23.1: @Foo.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc7_23.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_23.1
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
// CHECK:STDOUT:   %a.patt.loc6 => constants.%a.patt
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
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.1: type = generic_interface_type @Foo.loc6 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.1: %Foo.type.bc6a25.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bc6a25.2: type = generic_interface_type @Foo.loc10 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.2: %Foo.type.bc6a25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo.loc10, @Foo.loc10(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.bc6a25.1 = interface_decl @Foo.loc6 [concrete = constants.%Foo.generic.d3ea5a.1] {
// CHECK:STDOUT:     %a.patt.loc6_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc10: %Foo.type.bc6a25.2 = interface_decl @Foo.loc10 [concrete = constants.%Foo.generic.d3ea5a.2] {
// CHECK:STDOUT:     %a.patt.loc10_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc10_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc6(%a.loc6_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc6_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc10(%a.loc10_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc10_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc10_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc10, @Foo.loc10(%a.loc10_16.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc10_23.2: @Foo.loc10.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc10_23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc10_23.1: @Foo.loc10.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc10_23.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc10 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc10_23.1
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
// CHECK:STDOUT:   %a.patt.loc6_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc6_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc10(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc10_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc10_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.1: type = generic_interface_type @Foo.loc6 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.1: %Foo.type.bc6a25.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bc6a25.2: type = generic_interface_type @Foo.loc10 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.2: %Foo.type.bc6a25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo.loc10, @Foo.loc10(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.bc6a25.1 = interface_decl @Foo.loc6 [concrete = constants.%Foo.generic.d3ea5a.1] {
// CHECK:STDOUT:     %a.patt.loc6_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc10: %Foo.type.bc6a25.2 = interface_decl @Foo.loc10 [concrete = constants.%Foo.generic.d3ea5a.2] {
// CHECK:STDOUT:     %a.patt.loc10_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc10_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc6(%a.loc6_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc6_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc10(%a.loc10_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc10_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc10_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc10_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc10, @Foo.loc10(%a.loc10_16.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc10_23.2: @Foo.loc10.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc10_23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc10_23.1: @Foo.loc10.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc10_23.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc10 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc10_23.1
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
// CHECK:STDOUT:   %a.patt.loc6_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc6_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc10(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc10_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc10_16.1 => constants.%a
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
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:     %a.patt.loc7_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = interface_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc8_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc8_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc8_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(%a.loc8_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc8_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_16.2 (constants.%a.patt)]
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
// CHECK:STDOUT:   %a.patt.loc7_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc8_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc8_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.1: %Foo.type.bc6a25.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bc6a25.2: type = generic_interface_type @Foo.loc8 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.2: %Foo.type.bc6a25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo.loc8, @Foo.loc8(%a)> [symbolic]
// CHECK:STDOUT:   %Self.a95: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Bar.type.3f06c8.1: type = generic_interface_type @Bar.1 [concrete]
// CHECK:STDOUT:   %Bar.generic.b3011f.1: %Bar.type.3f06c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.3f06c8.2: type = generic_interface_type @Bar.loc13 [concrete]
// CHECK:STDOUT:   %Bar.generic.b3011f.2: %Bar.type.3f06c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type.ab7: type = facet_type <@Bar.loc13, @Bar.loc13(%a)> [symbolic]
// CHECK:STDOUT:   %Self.a90: %Bar.type.ab7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//two_file, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.Foo: %Foo.type.bc6a25.1 = import_ref Main//two_file, Foo, loaded [concrete = constants.%Foo.generic.d3ea5a.1]
// CHECK:STDOUT:   %Main.Bar: %Bar.type.3f06c8.1 = import_ref Main//two_file, Bar, loaded [concrete = constants.%Bar.generic.b3011f.1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.99696d.1: %C = import_ref Main//two_file, loc7_16, loaded [symbolic = @Foo.1.%a (constants.%a)]
// CHECK:STDOUT:   %Main.import_ref.99696d.2: %C = import_ref Main//two_file, loc8_16, loaded [symbolic = @Bar.1.%a (constants.%a)]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type.bc6a25.2 = interface_decl @Foo.loc8 [concrete = constants.%Foo.generic.d3ea5a.2] {
// CHECK:STDOUT:     %a.patt.loc8_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc8_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc8_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type.3f06c8.2 = interface_decl @Bar.loc13 [concrete = constants.%Bar.generic.b3011f.2] {
// CHECK:STDOUT:     %a.patt.loc13_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc13_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc13_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc13_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(imports.%Main.import_ref.99696d.1: %C) [from "two_file.carbon"] {
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc8(%a.loc8_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc8_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc8_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc8_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc8, @Foo.loc8(%a.loc8_16.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc8_21.2: @Foo.loc8.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc8_21.2 (constants.%Self.a95)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_21.1: @Foo.loc8.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc8_21.2 (constants.%Self.a95)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc8 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_21.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar.1(imports.%Main.import_ref.99696d.2: %C) [from "two_file.carbon"] {
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar.loc13(%a.loc13_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc13_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc13_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc13_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc13_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Bar.type: type = facet_type <@Bar.loc13, @Bar.loc13(%a.loc13_16.1)> [symbolic = %Bar.type (constants.%Bar.type.ab7)]
// CHECK:STDOUT:   %Self.loc13_21.2: @Bar.loc13.%Bar.type (%Bar.type.ab7) = symbolic_binding Self, 1 [symbolic = %Self.loc13_21.2 (constants.%Self.a90)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc13_21.1: @Bar.loc13.%Bar.type (%Bar.type.ab7) = symbolic_binding Self, 1 [symbolic = %Self.loc13_21.2 (constants.%Self.a90)]
// CHECK:STDOUT:     %Bar.WithSelf.decl = interface_with_self_decl @Bar.loc13 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc13_21.1
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.patt => constants.%a.patt
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc8(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc8_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc8_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self.a95) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.1(constants.%a) {
// CHECK:STDOUT:   %a.patt => constants.%a.patt
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.loc13(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc13_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc13_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar.WithSelf(constants.%a, constants.%Self.a90) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.1: type = generic_interface_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.1: %Foo.type.bc6a25.1 = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %b: %C = symbolic_binding b, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.2: type = generic_interface_type @Foo.loc11 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.2: %Foo.type.bc6a25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo.loc11, @Foo.loc11(%b)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.bc6a25.1 = interface_decl @Foo.loc7 [concrete = constants.%Foo.generic.d3ea5a.1] {
// CHECK:STDOUT:     %a.patt.loc7_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc11: %Foo.type.bc6a25.2 = interface_decl @Foo.loc11 [concrete = constants.%Foo.generic.d3ea5a.2] {
// CHECK:STDOUT:     %b.patt.loc11_16.1: %pattern_type = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc11_16.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc11_16.2: %C = symbolic_binding b, 0 [symbolic = %b.loc11_16.1 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc7(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc11(%b.loc11_16.2: %C) {
// CHECK:STDOUT:   %b.patt.loc11_16.2: %pattern_type = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc11_16.2 (constants.%b.patt)]
// CHECK:STDOUT:   %b.loc11_16.1: %C = symbolic_binding b, 0 [symbolic = %b.loc11_16.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc11, @Foo.loc11(%b.loc11_16.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc11_21.2: @Foo.loc11.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc11_21.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11_21.1: @Foo.loc11.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc11_21.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc11 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11_21.1
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
// CHECK:STDOUT:   %a.patt.loc7_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc11(constants.%b) {
// CHECK:STDOUT:   %b.patt.loc11_16.2 => constants.%b.patt
// CHECK:STDOUT:   %b.loc11_16.1 => constants.%b
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
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.1: type = generic_interface_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.1: %Foo.type.bc6a25.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bc6a25.2: type = generic_interface_type @Foo.loc11 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.2: %Foo.type.bc6a25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo.loc11, @Foo.loc11(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.bc6a25.1 = interface_decl @Foo.loc7 [concrete = constants.%Foo.generic.d3ea5a.1] {
// CHECK:STDOUT:     %a.patt.loc7_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc11: %Foo.type.bc6a25.2 = interface_decl @Foo.loc11 [concrete = constants.%Foo.generic.d3ea5a.2] {
// CHECK:STDOUT:     %a.patt.loc11_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc11_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc11_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc7(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc11(%a.loc11_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc11_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc11_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc11_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc11_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc11, @Foo.loc11(%a.loc11_16.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc11_21.2: @Foo.loc11.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc11_21.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11_21.1: @Foo.loc11.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc11_21.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc11 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11_21.1
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
// CHECK:STDOUT:   %a.patt.loc7_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc11(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc11_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc11_16.1 => constants.%a
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
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.1: type = generic_interface_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.1: %Foo.type.bc6a25.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bc6a25.2: type = generic_interface_type @Foo.loc11 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.2: %Foo.type.bc6a25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo.loc11, @Foo.loc11(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.bc6a25.1 = interface_decl @Foo.loc7 [concrete = constants.%Foo.generic.d3ea5a.1] {
// CHECK:STDOUT:     %a.patt.loc7_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc11: %Foo.type.bc6a25.2 = interface_decl @Foo.loc11 [concrete = constants.%Foo.generic.d3ea5a.2] {
// CHECK:STDOUT:     %a.patt.loc11_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc11_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc11_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc7(%a.loc7_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc11(%a.loc11_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc11_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc11_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc11_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc11_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc11, @Foo.loc11(%a.loc11_16.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc11_23.2: @Foo.loc11.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc11_23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11_23.1: @Foo.loc11.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc11_23.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc11 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11_23.1
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
// CHECK:STDOUT:   %a.patt.loc7_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc11(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc11_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc11_16.1 => constants.%a
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
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:     %a.patt.loc6_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(%a.loc6_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc6_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt)]
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
// CHECK:STDOUT:   %a.patt.loc6_16.2 => constants.%a.patt
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
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.1: type = generic_interface_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.1: %Foo.type.bc6a25.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bc6a25.2: type = generic_interface_type @Foo.loc13 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.2: %Foo.type.bc6a25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.8ee: type = facet_type <@Foo.loc13, @Foo.loc13(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.8ee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//alias_two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.Foo: %Foo.type.bc6a25.1 = import_ref Main//alias_two_file, Foo, loaded [concrete = constants.%Foo.generic.d3ea5a.1]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//alias_two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//alias_two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.996: %C = import_ref Main//alias_two_file, loc6_16, loaded [symbolic = @Foo.1.%a (constants.%a)]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type.bc6a25.2 = interface_decl @Foo.loc13 [concrete = constants.%Foo.generic.d3ea5a.2] {
// CHECK:STDOUT:     %a.patt.loc13_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc13_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc13_16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc13_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.1(imports.%Main.import_ref.996: %C) [from "alias_two_file.carbon"] {
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc13(%a.loc13_16.2: %C) {
// CHECK:STDOUT:   %a.patt.loc13_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc13_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc13_16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc13_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc13, @Foo.loc13(%a.loc13_16.1)> [symbolic = %Foo.type (constants.%Foo.type.8ee)]
// CHECK:STDOUT:   %Self.loc13_21.2: @Foo.loc13.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc13_21.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc13_21.1: @Foo.loc13.%Foo.type (%Foo.type.8ee) = symbolic_binding Self, 1 [symbolic = %Self.loc13_21.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc13 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc13_21.1
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.patt => constants.%a.patt
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc13(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc13_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc13_16.1 => constants.%a
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
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %const = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.bc6a25.1: type = generic_interface_type @Foo.loc6 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.1: %Foo.type.bc6a25.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.bc6a25.2: type = generic_interface_type @Foo.loc12 [concrete]
// CHECK:STDOUT:   %Foo.generic.d3ea5a.2: %Foo.type.bc6a25.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.b25: type = facet_type <@Foo.loc12, @Foo.loc12(%a)> [symbolic]
// CHECK:STDOUT:   %Self: %Foo.type.b25 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.bc6a25.1 = interface_decl @Foo.loc6 [concrete = constants.%Foo.generic.d3ea5a.1] {
// CHECK:STDOUT:     %a.patt.loc6_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_16.2: %const = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc12: %Foo.type.bc6a25.2 = interface_decl @Foo.loc12 [concrete = constants.%Foo.generic.d3ea5a.2] {
// CHECK:STDOUT:     %a.patt.loc12_16.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc12_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12: type = splice_block %const.loc12_18 [concrete = constants.%const] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc12_25: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:       %const.loc12_18: type = const_type %const.loc12_25 [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc12_16.2: %const = symbolic_binding a, 0 [symbolic = %a.loc12_16.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc6(%a.loc6_16.2: %const) {
// CHECK:STDOUT:   %a.patt.loc6_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_16.1: %const = symbolic_binding a, 0 [symbolic = %a.loc6_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo.loc12(%a.loc12_16.2: %const) {
// CHECK:STDOUT:   %a.patt.loc12_16.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc12_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc12_16.1: %const = symbolic_binding a, 0 [symbolic = %a.loc12_16.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Foo.type: type = facet_type <@Foo.loc12, @Foo.loc12(%a.loc12_16.1)> [symbolic = %Foo.type (constants.%Foo.type.b25)]
// CHECK:STDOUT:   %Self.loc12_35.2: @Foo.loc12.%Foo.type (%Foo.type.b25) = symbolic_binding Self, 1 [symbolic = %Self.loc12_35.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc12_35.1: @Foo.loc12.%Foo.type (%Foo.type.b25) = symbolic_binding Self, 1 [symbolic = %Self.loc12_35.2 (constants.%Self)]
// CHECK:STDOUT:     %Foo.WithSelf.decl = interface_with_self_decl @Foo.loc12 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc12_35.1
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
// CHECK:STDOUT:   %a.patt.loc6_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc6_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc12(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc12_16.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc12_16.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.WithSelf(constants.%a, constants.%Self) {}
// CHECK:STDOUT:
