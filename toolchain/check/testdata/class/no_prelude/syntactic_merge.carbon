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

  fn F[addr self: Self*]();
}

// CHECK:STDERR: fail_self_type.carbon:[[@LINE+6]]:22: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Base.F[addr self: Base*]() {
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR: fail_self_type.carbon:[[@LINE-6]]:19: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR:   fn F[addr self: Self*]();
// CHECK:STDERR:                   ^~~~
fn Base.F[addr self: Base*]() {
  (*self).a = ();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Foo.2: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT:   %Bar.type: type = generic_class_type @Bar [template]
// CHECK:STDOUT:   %Bar.1: %Bar.type = struct_value () [template]
// CHECK:STDOUT:   %Bar.2: type = class_type @Bar, @Bar(%a) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc8: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc8, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0, %a.param.loc7 [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc8: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc8, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc8: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc8: %C = bind_symbolic_name a, 0, %a.param.loc8 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type = class_decl @Bar [template = constants.%Bar.1] {
// CHECK:STDOUT:     %a.patt.loc11: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc11, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc10: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc10_11.1: %C = bind_symbolic_name a, 0, %a.param.loc10 [symbolic = %a.loc10_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type = class_decl @Bar [template = constants.%Bar.1] {
// CHECK:STDOUT:     %a.patt.loc11: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc11, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc11: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc11: %C = bind_symbolic_name a, 0, %a.param.loc11 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT:     %.loc8: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2
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
// CHECK:STDOUT:     %.loc11: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Bar.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc10_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc10 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Foo.2: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc6: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_17.1: %C = bind_symbolic_name a, 0, %a.param.loc6 [symbolic = %a.loc6_17.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0, %a.param.loc7 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT:     %.loc7: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_17.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = class_type @.1, @.1(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc6_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6_11.1, runtime_param<invalid> [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %a.patt.loc14_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc14_11.1, runtime_param<invalid> [symbolic = %a.patt.loc14_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc14_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc14_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT: generic class @.1(%a.loc14_11.1: %C) {
// CHECK:STDOUT:   %a.loc14_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc14_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc14_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc14_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc14: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc14_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc14_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Foo.2: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc6: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_11.1: %C = bind_symbolic_name a, 0, %a.param.loc6 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc7: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0, %a.param.loc7 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT:     %.loc7: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Bar.type: type = generic_class_type @Bar [template]
// CHECK:STDOUT:   %Bar.1: %Bar.type = struct_value () [template]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc7_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_11.1, runtime_param<invalid> [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = class_decl @Bar [template = constants.%Bar.1] {
// CHECK:STDOUT:     %a.patt.loc8_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc8_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc8_11.1, runtime_param<invalid> [symbolic = %a.patt.loc8_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc8_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc8_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT:   %a.patt.loc7_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc8_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc8_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Foo.2: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT:   %Bar.type: type = generic_class_type @Bar [template]
// CHECK:STDOUT:   %Bar.1: %Bar.type = struct_value () [template]
// CHECK:STDOUT:   %Bar.2: type = class_type @Bar, @Bar(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//two_file, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//two_file, inst+7, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//two_file, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .D = imports.%import_ref.2
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc4: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc4, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc4: %C = bind_symbolic_name a, 0, %a.param [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = class_decl @Bar [template = constants.%Bar.1] {
// CHECK:STDOUT:     %a.patt.loc5: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc5, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.2 [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc5: %C = bind_symbolic_name a, 0, %a.param [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo(constants.%a: %C) {
// CHECK:STDOUT:   %a.1: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   %a.patt.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(constants.%a: %C) {
// CHECK:STDOUT:   %a.1: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   %a.patt.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Bar.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT:   %a.patt.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT:   %a.patt.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %b: %C = bind_symbolic_name b, 0 [symbolic]
// CHECK:STDOUT:   %b.patt: %C = symbolic_binding_pattern b, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = class_type @.1, @.1(%b) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc7_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_11.1, runtime_param<invalid> [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %b.patt.loc15_11.1: %C = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_11.2 (constants.%b.patt)]
// CHECK:STDOUT:     %b.param_patt: %C = value_param_pattern %b.patt.loc15_11.1, runtime_param<invalid> [symbolic = %b.patt.loc15_11.2 (constants.%b.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %b.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %b.loc15_11.1: %C = bind_symbolic_name b, 0, %b.param [symbolic = %b.loc15_11.2 (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT: generic class @.1(%b.loc15_11.1: %C) {
// CHECK:STDOUT:   %b.loc15_11.2: %C = bind_symbolic_name b, 0 [symbolic = %b.loc15_11.2 (constants.%b)]
// CHECK:STDOUT:   %b.patt.loc15_11.2: %C = symbolic_binding_pattern b, 0 [symbolic = %b.patt.loc15_11.2 (constants.%b.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc15: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%b) {
// CHECK:STDOUT:   %b.loc15_11.2 => constants.%b
// CHECK:STDOUT:   %b.patt.loc15_11.2 => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = class_type @.1, @.1(%a) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc7_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_11.1, runtime_param<invalid> [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %a.patt.loc15_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc15_11.1, runtime_param<invalid> [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc15_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc15_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT: generic class @.1(%a.loc15_11.1: %C) {
// CHECK:STDOUT:   %a.loc15_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc15_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc15: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc15_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduced_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = class_type @.1, @.1(%a) [symbolic]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc7_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc7_11.1, runtime_param<invalid> [symbolic = %a.patt.loc7_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc7_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %a.patt.loc15_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc15_11.1, runtime_param<invalid> [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc15_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc15_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT: generic class @.1(%a.loc15_11.1: %C) {
// CHECK:STDOUT:   %a.loc15_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc15_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc15_11.2: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc15_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc15: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc7_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc15_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc15_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc6_11.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6_11.1, runtime_param<invalid> [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_11.1: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT:   %a.patt.loc6_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %C = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Foo.2: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//alias_two_file, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//alias_two_file, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc6: %C = symbolic_binding_pattern a, 0 [symbolic = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %C = value_param_pattern %a.patt.loc6, runtime_param<invalid> [symbolic = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6: %C = bind_symbolic_name a, 0, %a.param [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo(constants.%a: %C) {
// CHECK:STDOUT:   %a.1: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   %a.patt.1: %C = symbolic_binding_pattern a, 0 [symbolic = %a.patt.1 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT:   %a.patt.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeat_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = const_type %C [template]
// CHECK:STDOUT:   %a: %.3 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %.3 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [template]
// CHECK:STDOUT:   %Foo.1: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = class_type @.1, @.1(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [template = constants.%Foo.1] {
// CHECK:STDOUT:     %a.patt.loc6_11.1: %.3 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %.3 = value_param_pattern %a.patt.loc6_11.1, runtime_param<invalid> [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc6: type = const_type %C [template = constants.%.3]
// CHECK:STDOUT:     %a.param: %.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_11.1: %.3 = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %a.patt.loc18_11.1: %.3 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc18_11.2 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %.3 = value_param_pattern %a.patt.loc18_11.1, runtime_param<invalid> [symbolic = %a.patt.loc18_11.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc18_22: type = const_type %C [template = constants.%.3]
// CHECK:STDOUT:     %.loc18_15: type = const_type %.3 [template = constants.%.3]
// CHECK:STDOUT:     %a.param: %.3 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc18_11.1: %.3 = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc18_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo(%a.loc6_11.1: %.3) {
// CHECK:STDOUT:   %a.loc6_11.2: %.3 = bind_symbolic_name a, 0 [symbolic = %a.loc6_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc6_11.2: %.3 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc6_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%a.loc18_11.1: %.3) {
// CHECK:STDOUT:   %a.loc18_11.2: %.3 = bind_symbolic_name a, 0 [symbolic = %a.loc18_11.2 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc18_11.2: %.3 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc18_11.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc18_33: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc6_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc6_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.loc18_11.2 => constants.%a
// CHECK:STDOUT:   %a.patt.loc18_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Base, %empty_tuple.type [template]
// CHECK:STDOUT:   %.2: type = ptr_type %Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %empty_tuple.type} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.5: %.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.5] {
// CHECK:STDOUT:     %self.patt: %.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc16_11: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc16_26: type = ptr_type %Base [template = constants.%.2]
// CHECK:STDOUT:     %self.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc5_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_11.2: type = converted %.loc5_11.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc5_8: %.1 = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc7_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %.loc7_23: type = ptr_type %Base [template = constants.%.2]
// CHECK:STDOUT:     %self.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.param_patt: %.2]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1[addr %self.param_patt: %.2]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.2 = name_ref self, %self
// CHECK:STDOUT:   %.loc17_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, @Base.%.loc5_8 [template = @Base.%.loc5_8]
// CHECK:STDOUT:   %.loc17_10: ref %empty_tuple.type = class_element_access %.loc17_4, element0
// CHECK:STDOUT:   %.loc17_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc17_16.2: init %empty_tuple.type = tuple_init () to %.loc17_10 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_13: init %empty_tuple.type = converted %.loc17_16.1, %.loc17_16.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc17_10, %.loc17_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
