// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/syntactic_merge.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/syntactic_merge.carbon

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

// --- todo_fail_raw_identifier.carbon

library "[[@TEST_NAME]]";

class C {}

class Foo(a:! C);
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
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-4]]:11: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Foo(a:! C);
// CHECK:STDERR:           ^
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

  fn F[addr self: Self*]();
}

// CHECK:STDERR: fail_self_type.carbon:[[@LINE+7]]:22: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Base.F[addr self: Base*]() {
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR: fail_self_type.carbon:[[@LINE-6]]:19: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:   fn F[addr self: Self*]();
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR:
fn Base.F[addr self: Base*]() {
  (*self).a = ();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc8: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc8: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc8: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc11: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc10_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc10_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc11: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc11: %C = bind_symbolic_name a, 0 [symbolic = %a.loc10_11.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc7_11.1: %C) {
// CHECK:STDOUT:   %a.loc7_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(%a.loc10_11.1: %C) {
// CHECK:STDOUT:   %a.loc10_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc10_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc10: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc10 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc10_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc10 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:     %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6_17.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_17.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_17.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc6_17.1: %C) {
// CHECK:STDOUT:   %a.loc6_17.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_17.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_17.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.39e446.1: type = generic_class_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.1: %Foo.type.39e446.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.39e446.2: type = generic_class_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.2: %Foo.type.39e446.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.67ab7c.2: type = class_type @Foo.2, @Foo.2(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.39e446.1 = class_decl @Foo.1 [concrete = constants.%Foo.generic.80461b.1] {
// CHECK:STDOUT:     %a.patt.loc6_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc14: %Foo.type.39e446.2 = class_decl @Foo.2 [concrete = constants.%Foo.generic.80461b.2] {
// CHECK:STDOUT:     %a.patt.loc14_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc14_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc14_11.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.1(%a.loc6_11.1: %C) {
// CHECK:STDOUT:   %a.loc6_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.2(%a.loc14_11.1: %C) {
// CHECK:STDOUT:   %a.loc14_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc14_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc14_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.67ab7c.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc6_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc14_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc14_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:     %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc6_11.1: %C) {
// CHECK:STDOUT:   %a.loc6_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc7_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc8_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc8_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc8_11.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc7_11.1: %C) {
// CHECK:STDOUT:   %a.loc7_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(%a.loc8_11.1: %C) {
// CHECK:STDOUT:   %a.loc8_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc8_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc8_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc8_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc8_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//two_file, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f97b44.1: %C = import_ref Main//two_file, loc7_11, loaded [symbolic = @Foo.%a.1 (constants.%a)]
// CHECK:STDOUT:   %Main.import_ref.f97b44.2: %C = import_ref Main//two_file, loc8_11, loaded [symbolic = @Bar.%a.1 (constants.%a)]
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
// CHECK:STDOUT:     %a.patt.loc4: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc4: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt.loc5: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc5: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(imports.%Main.import_ref.f97b44.1: %C) {
// CHECK:STDOUT:   %a.1: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   %a.patt.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(imports.%Main.import_ref.f97b44.2: %C) {
// CHECK:STDOUT:   %a.1: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   %a.patt.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT:   %a.patt.1 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT:   %a.patt.1 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.39e446.1: type = generic_class_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.1: %Foo.type.39e446.1 = struct_value () [concrete]
// CHECK:STDOUT:   %b: %C = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.39e446.2: type = generic_class_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.2: %Foo.type.39e446.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.67ab7c.2: type = class_type @Foo.2, @Foo.2(%b) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.39e446.1 = class_decl @Foo.1 [concrete = constants.%Foo.generic.80461b.1] {
// CHECK:STDOUT:     %a.patt.loc7_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.39e446.2 = class_decl @Foo.2 [concrete = constants.%Foo.generic.80461b.2] {
// CHECK:STDOUT:     %b.patt.loc15_11.1: %C = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_11.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %b.loc15_11.1: %C = bind_symbolic_name b, 0 [symbolic = %b.loc15_11.2 (constants.%b)]
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
// CHECK:STDOUT: generic class @Foo.1(%a.loc7_11.1: %C) {
// CHECK:STDOUT:   %a.loc7_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.2(%b.loc15_11.1: %C) {
// CHECK:STDOUT:   %b.loc15_11.2: %C = bind_symbolic_name b, 0 [symbolic = %b.loc15_11.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc15_11.2: %C = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_11.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.67ab7c.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%b) {
// CHECK:STDOUT:   %b.loc15_11.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc15_11.2 => constants.%b.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.39e446.1: type = generic_class_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.1: %Foo.type.39e446.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.39e446.2: type = generic_class_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.2: %Foo.type.39e446.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.67ab7c.2: type = class_type @Foo.2, @Foo.2(%a) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.39e446.1 = class_decl @Foo.1 [concrete = constants.%Foo.generic.80461b.1] {
// CHECK:STDOUT:     %a.patt.loc7_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.39e446.2 = class_decl @Foo.2 [concrete = constants.%Foo.generic.80461b.2] {
// CHECK:STDOUT:     %a.patt.loc15_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc15_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_11.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.1(%a.loc7_11.1: %C) {
// CHECK:STDOUT:   %a.loc7_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.2(%a.loc15_11.1: %C) {
// CHECK:STDOUT:   %a.loc15_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc15_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.67ab7c.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc15_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduced_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.39e446.1: type = generic_class_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.1: %Foo.type.39e446.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.39e446.2: type = generic_class_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.2: %Foo.type.39e446.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.67ab7c.2: type = class_type @Foo.2, @Foo.2(%a) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type.39e446.1 = class_decl @Foo.1 [concrete = constants.%Foo.generic.80461b.1] {
// CHECK:STDOUT:     %a.patt.loc7_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.39e446.2 = class_decl @Foo.2 [concrete = constants.%Foo.generic.80461b.2] {
// CHECK:STDOUT:     %a.patt.loc15_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc15_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_11.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.1(%a.loc7_11.1: %C) {
// CHECK:STDOUT:   %a.loc7_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc7_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.2(%a.loc15_11.1: %C) {
// CHECK:STDOUT:   %a.loc15_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc15_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.67ab7c.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc15_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
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
// CHECK:STDOUT:     %a.patt.loc6_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(%a.loc6_11.1: %C) {
// CHECK:STDOUT:   %a.loc6_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Foo: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//alias_two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//alias_two_file, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//alias_two_file, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f97: %C = import_ref Main//alias_two_file, loc6_11, loaded [symbolic = @Foo.%a.1 (constants.%a)]
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
// CHECK:STDOUT:   %D: type = bind_alias D, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt.loc6: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo(imports.%Main.import_ref.f97: %C) {
// CHECK:STDOUT:   %a.1: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   %a.patt.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT:   %a.patt.1 => constants.%a.patt
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Foo.type.39e446.1: type = generic_class_type @Foo.1 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.1: %Foo.type.39e446.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.39e446.2: type = generic_class_type @Foo.2 [concrete]
// CHECK:STDOUT:   %Foo.generic.80461b.2: %Foo.type.39e446.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.1a4f0e.2: type = class_type @Foo.2, @Foo.2(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type.39e446.1 = class_decl @Foo.1 [concrete = constants.%Foo.generic.80461b.1] {
// CHECK:STDOUT:     %a.patt.loc6_11.1: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc6_11.1: %const = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc18: %Foo.type.39e446.2 = class_decl @Foo.2 [concrete = constants.%Foo.generic.80461b.2] {
// CHECK:STDOUT:     %a.patt.loc18_11.1: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc18_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %const.loc18_15 [concrete = constants.%const] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc18_22: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:       %const.loc18_15: type = const_type %const.loc18_22 [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc18_11.1: %const = bind_symbolic_name a, 0 [symbolic = %a.loc18_11.2 (constants.%a)]
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
// CHECK:STDOUT: generic class @Foo.1(%a.loc6_11.1: %const) {
// CHECK:STDOUT:   %a.loc6_11.2: %const = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_11.2: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo.2(%a.loc18_11.1: %const) {
// CHECK:STDOUT:   %a.loc18_11.2: %const = bind_symbolic_name a, 0 [symbolic = %a.loc18_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc18_11.2: %const = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc18_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.1a4f0e.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.1(constants.%a) {
// CHECK:STDOUT:   %a.loc6_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.2(constants.%a) {
// CHECK:STDOUT:   %a.loc18_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc18_11.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %F.type.7c6384.1: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.d17bbc.1: %F.type.7c6384.1 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT:   %F.type.7c6384.2: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.d17bbc.2: %F.type.7c6384.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.7c6384.2 = fn_decl @F.2 [concrete = constants.%F.d17bbc.2] {
// CHECK:STDOUT:     %self.patt: %ptr.11f = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.11f = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc17_11: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %.loc17_26: type = splice_block %ptr [concrete = constants.%ptr.11f] {
// CHECK:STDOUT:       %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:       %ptr: type = ptr_type %Base.ref [concrete = constants.%ptr.11f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.11f = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc5_8: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %Base.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Base.elem = var <none>
// CHECK:STDOUT:   %F.decl: %F.type.7c6384.1 = fn_decl @F.1 [concrete = constants.%F.d17bbc.1] {
// CHECK:STDOUT:     %self.patt: %ptr.11f = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.11f = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc7_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %.loc7_23: type = splice_block %ptr [concrete = constants.%ptr.11f] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.11f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.11f = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self.param_patt: %ptr.11f]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[addr %self.param_patt: %ptr.11f]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.11f = name_ref self, %self
// CHECK:STDOUT:   %.loc18_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc5_8 [concrete = @Base.%.loc5_8]
// CHECK:STDOUT:   %.loc18_10: ref %empty_tuple.type = class_element_access %.loc18_4, element0
// CHECK:STDOUT:   %.loc18_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc18_16.2: init %empty_tuple.type = tuple_init () to %.loc18_10 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_13: init %empty_tuple.type = converted %.loc18_16.1, %.loc18_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc18_10, %.loc18_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
