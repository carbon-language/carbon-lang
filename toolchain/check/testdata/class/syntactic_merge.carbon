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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/syntactic_merge.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/syntactic_merge.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo(a:! C);
class Foo(a:! C) {}

class Bar(a:! D);
class Bar(a:! D) {}

// --- spacing.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo [ ] ( a :! C );
class Foo[](a:! C) {}

// --- fail_parens.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! C);
// CHECK:STDERR: fail_parens.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo(a:! (C)) {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_parens.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo(a:! C);
// CHECK:STDERR:               ^
// CHECK:STDERR:
class Foo(a:! (C)) {}

// --- fail_raw_identifier.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! C);
// CHECK:STDERR: fail_raw_identifier.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo(a:! r#C) {}
// CHECK:STDERR:               ^~~
// CHECK:STDERR: fail_raw_identifier.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo(a:! C);
// CHECK:STDERR:               ^
// CHECK:STDERR:
class Foo(a:! r#C) {}

// --- two_file.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo(a:! C);
class Bar(a:! D);

// --- two_file.impl.carbon

impl library "[[@TEST_NAME]]";

class Foo(a:! C) {}
class Bar(a:! D) {}

// --- fail_name_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo(a:! C);
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE+7]]:11: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: class Foo(b:! D) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-4]]:11: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Foo(a:! C);
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
class Foo(b:! D) {}

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo(a:! C);
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo(a:! D) {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo(a:! C);
// CHECK:STDERR:               ^
// CHECK:STDERR:
class Foo(a:! D) {}

// --- fail_deduced_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

class Foo[a:! C]();
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE+7]]:15: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo[a:! D]() {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE-4]]:15: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo[a:! C]();
// CHECK:STDERR:               ^
// CHECK:STDERR:
class Foo[a:! D]() {}

// --- alias_two_file.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! C);

// --- todo_fail_alias_two_file.impl.carbon

impl library "[[@TEST_NAME]]";

alias D = C;

class Foo(a:! D) {}

// --- fail_repeat_const.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! const C);
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+11]]:15: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: class Foo(a:! const (const C)) {}
// CHECK:STDERR:               ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+7]]:21: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: class Foo(a:! const (const C)) {}
// CHECK:STDERR:                     ^
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE-8]]:21: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: class Foo(a:! const C);
// CHECK:STDERR:                     ^
// CHECK:STDERR:
class Foo(a:! const (const C)) {}

// --- fail_self_type.carbon

library "[[@TEST_NAME]]";

base class Base {
  var a: ();

  fn F(ref self);
}

// CHECK:STDERR: fail_self_type.carbon:[[@LINE+7]]:21: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Base.F(ref self: Base) {
// CHECK:STDERR:                     ^~~~
// CHECK:STDERR: fail_self_type.carbon:[[@LINE-6]]:8: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:   fn F(ref self);
// CHECK:STDERR:        ^~~~~~~~
// CHECK:STDERR:
fn Base.F(ref self: Base) {
  self.a = ();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Foo: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT:   %Bar.type: type = generic_class_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar, @Bar(%a) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc8: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref.loc7 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc8: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %C.ref.loc8 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc8: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc8: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc11: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %D.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref.loc10: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc10_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc10_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc11: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %D.ref.loc11 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc11: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref.loc11: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc11: %C = symbolic_binding a, 0 [symbolic = %a.loc10_12.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc7_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(%a.loc10_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc10: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc10_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc10_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc7 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc10 => constants.%a.patt
// CHECK:STDOUT:   %a.loc10_12.1 => constants.%a
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Foo: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref.loc6 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc6: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_19.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_19.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref.loc7 [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.loc7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7: %C = symbolic_binding a, 0 [symbolic = %a.loc6_19.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc6_19.2: %C) {
// CHECK:STDOUT:   %a.patt.loc6: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_19.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_19.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc6 => constants.%a.patt
// CHECK:STDOUT:   %a.loc6_19.1 => constants.%a
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.3e04c8.1: type = generic_class_type @Foo.loc6 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.1: %Foo.type.3e04c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.3e04c8.2: type = generic_class_type @Foo.loc14 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.2: %Foo.type.3e04c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.2fb251.2: type = class_type @Foo.loc14, @Foo.loc14(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.3e04c8.1 = class_decl @Foo.loc6 [concrete = constants.%Foo.generic.04949d.1] {
// CHECK:STDOUT:     %a.patt.loc6_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc14: %Foo.type.3e04c8.2 = class_decl @Foo.loc14 [concrete = constants.%Foo.generic.04949d.2] {
// CHECK:STDOUT:     %a.patt.loc14_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc14_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc14_12.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.loc6(%a.loc6_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc6_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.loc14(%a.loc14_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc14_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc14_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc14_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2fb251.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc6(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc6_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc6_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc14(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc14_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc14_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.3e04c8.1: type = generic_class_type @Foo.loc6 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.1: %Foo.type.3e04c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.3e04c8.2: type = generic_class_type @Foo.loc14 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.2: %Foo.type.3e04c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.2fb251.2: type = class_type @Foo.loc14, @Foo.loc14(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.3e04c8.1 = class_decl @Foo.loc6 [concrete = constants.%Foo.generic.04949d.1] {
// CHECK:STDOUT:     %a.patt.loc6_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc14: %Foo.type.3e04c8.2 = class_decl @Foo.loc14 [concrete = constants.%Foo.generic.04949d.2] {
// CHECK:STDOUT:     %a.patt.loc14_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc14_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc14_12.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.loc6(%a.loc6_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc6_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.loc14(%a.loc14_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc14_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc14_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc14_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2fb251.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc6(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc6_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc6_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc14(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc14_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc14_12.1 => constants.%a
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type: type = generic_class_type @Bar [concrete]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc8_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc8_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc8_12.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc7_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(%a.loc8_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc8_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc8_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc8_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc7_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc8_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc8_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Foo: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT:   %Bar.type: type = generic_class_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar, @Bar(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//two_file, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.99696d.1: %C = import_ref Main//two_file, loc7_12, loaded [symbolic = @Foo.%a.1 (constants.%a)]
// CHECK:STDOUT:   %Main.import_ref.99696d.2: %C = import_ref Main//two_file, loc8_12, loaded [symbolic = @Bar.%a.1 (constants.%a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_24.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_24.2 = import <none>
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc4: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc4: %C = symbolic_binding a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc5: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc5: %C = symbolic_binding a, 0 [symbolic = %a.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(imports.%Main.import_ref.99696d.1: %C) {
// CHECK:STDOUT:   %a.patt.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   %a.1: %C = symbolic_binding a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(imports.%Main.import_ref.99696d.2: %C) {
// CHECK:STDOUT:   %a.patt.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   %a.1: %C = symbolic_binding a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.patt.1 => constants.%a.patt
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.patt.1 => constants.%a.patt
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.3e04c8.1: type = generic_class_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.1: %Foo.type.3e04c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %b: %C = symbolic_binding b, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.3e04c8.2: type = generic_class_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.2: %Foo.type.3e04c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.2fb251.2: type = class_type @Foo.loc15, @Foo.loc15(%b) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.3e04c8.1 = class_decl @Foo.loc7 [concrete = constants.%Foo.generic.04949d.1] {
// CHECK:STDOUT:     %a.patt.loc7_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.3e04c8.2 = class_decl @Foo.loc15 [concrete = constants.%Foo.generic.04949d.2] {
// CHECK:STDOUT:     %b.patt.loc15_12.1: %pattern_type = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_12.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc15_12.2: %C = symbolic_binding b, 0 [symbolic = %b.loc15_12.1 (constants.%b)]
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
// CHECK:STDOUT: generic class @Foo.loc7(%a.loc7_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.loc15(%b.loc15_12.2: %C) {
// CHECK:STDOUT:   %b.patt.loc15_12.2: %pattern_type = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_12.2 (constants.%b.patt)]
// CHECK:STDOUT:   %b.loc15_12.1: %C = symbolic_binding b, 0 [symbolic = %b.loc15_12.1 (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2fb251.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc7(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc7_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc15(constants.%b) {
// CHECK:STDOUT:   %b.patt.loc15_12.2 => constants.%b.patt
// CHECK:STDOUT:   %b.loc15_12.1 => constants.%b
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.3e04c8.1: type = generic_class_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.1: %Foo.type.3e04c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.3e04c8.2: type = generic_class_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.2: %Foo.type.3e04c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.2fb251.2: type = class_type @Foo.loc15, @Foo.loc15(%a) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.3e04c8.1 = class_decl @Foo.loc7 [concrete = constants.%Foo.generic.04949d.1] {
// CHECK:STDOUT:     %a.patt.loc7_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.3e04c8.2 = class_decl @Foo.loc15 [concrete = constants.%Foo.generic.04949d.2] {
// CHECK:STDOUT:     %a.patt.loc15_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc15_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc15_12.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.loc7(%a.loc7_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.loc15(%a.loc15_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc15_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc15_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc15_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2fb251.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc7(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc7_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc15(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc15_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc15_12.1 => constants.%a
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.3e04c8.1: type = generic_class_type @Foo.loc7 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.1: %Foo.type.3e04c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.3e04c8.2: type = generic_class_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.2: %Foo.type.3e04c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.2fb251.2: type = class_type @Foo.loc15, @Foo.loc15(%a) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.3e04c8.1 = class_decl @Foo.loc7 [concrete = constants.%Foo.generic.04949d.1] {
// CHECK:STDOUT:     %a.patt.loc7_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc7_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.3e04c8.2 = class_decl @Foo.loc15 [concrete = constants.%Foo.generic.04949d.2] {
// CHECK:STDOUT:     %a.patt.loc15_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc15_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc15_12.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.loc7(%a.loc7_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc7_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc7_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc7_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.loc15(%a.loc15_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc15_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc15_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc15_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2fb251.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc7(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc7_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc7_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc15(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc15_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc15_12.1 => constants.%a
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_12.2: %C = symbolic_binding a, 0 [symbolic = %a.loc6_12.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc6_12.2: %C) {
// CHECK:STDOUT:   %a.patt.loc6_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_12.1: %C = symbolic_binding a, 0 [symbolic = %a.loc6_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc6_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc6_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Foo: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//alias_two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//alias_two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//alias_two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.996: %C = import_ref Main//alias_two_file, loc6_12, loaded [symbolic = @Foo.%a.1 (constants.%a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_30.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_30.2 = import <none>
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.ref [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6: %C = symbolic_binding a, 0 [symbolic = %a.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(imports.%Main.import_ref.996: %C) {
// CHECK:STDOUT:   %a.patt.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   %a.1: %C = symbolic_binding a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.patt.1 => constants.%a.patt
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %a: %const = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.3e04c8.1: type = generic_class_type @Foo.loc6 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.1: %Foo.type.3e04c8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.3e04c8.2: type = generic_class_type @Foo.loc18 [concrete]
// CHECK:STDOUT:   %Foo.generic.04949d.2: %Foo.type.3e04c8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.13c14a.2: type = class_type @Foo.loc18, @Foo.loc18(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.3e04c8.1 = class_decl @Foo.loc6 [concrete = constants.%Foo.generic.04949d.1] {
// CHECK:STDOUT:     %a.patt.loc6_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_12.2: %const = symbolic_binding a, 0 [symbolic = %a.loc6_12.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc18: %Foo.type.3e04c8.2 = class_decl @Foo.loc18 [concrete = constants.%Foo.generic.04949d.2] {
// CHECK:STDOUT:     %a.patt.loc18_12.1: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc18_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %const.loc18_15 [concrete = constants.%const] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc18_22: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:       %const.loc18_15: type = const_type %const.loc18_22 [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc18_12.2: %const = symbolic_binding a, 0 [symbolic = %a.loc18_12.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.loc6(%a.loc6_12.2: %const) {
// CHECK:STDOUT:   %a.patt.loc6_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc6_12.1: %const = symbolic_binding a, 0 [symbolic = %a.loc6_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.loc18(%a.loc18_12.2: %const) {
// CHECK:STDOUT:   %a.patt.loc18_12.2: %pattern_type = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc18_12.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.loc18_12.1: %const = symbolic_binding a, 0 [symbolic = %a.loc18_12.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.13c14a.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc6(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc6_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc6_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc18(constants.%a) {
// CHECK:STDOUT:   %a.patt.loc18_12.2 => constants.%a.patt
// CHECK:STDOUT:   %a.loc18_12.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.912: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.912 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type.912 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   %Base.F.type.48cdab.1: type = fn_type @Base.F.loc7 [concrete]
// CHECK:STDOUT:   %Base.F.bd5be1.1: %Base.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT:   %Base.F.type.48cdab.2: type = fn_type @Base.F.loc17 [concrete]
// CHECK:STDOUT:   %Base.F.bd5be1.2: %Base.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type.48cdab.2 = fn_decl @Base.F.loc17 [concrete = constants.%Base.F.bd5be1.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.912 = ref_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type.912 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %self: ref %Base = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc5: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type.48cdab.1 = fn_decl @Base.F.loc7 [concrete = constants.%Base.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.912 = ref_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type.912 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: ref %Base = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F.loc7(%self.param: ref %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F.loc17(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: ref %Base = name_ref self, %self
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:   %.loc18_7: ref %empty_tuple.type = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc18_13.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_13.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_10: init %empty_tuple.type = converted %.loc18_13.1, %.loc18_13.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc18_7, %.loc18_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
