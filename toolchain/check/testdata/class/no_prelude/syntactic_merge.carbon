// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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

//@dump-sem-ir-begin
class Foo(a:! C);
class Foo(a:! C) {}

class Bar(a:! D);
class Bar(a:! D) {}
//@dump-sem-ir-end

// --- spacing.carbon

library "[[@TEST_NAME]]";

class C {}

//@dump-sem-ir-begin
class Foo [ ] ( a :! C );
class Foo[](a:! C) {}
//@dump-sem-ir-end

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

//@dump-sem-ir-begin
class Foo(a:! C) {}
class Bar(a:! D) {}
//@dump-sem-ir-end

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
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Foo: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT:   %Bar.type: type = generic_class_type @Bar [concrete]
// CHECK:STDOUT:   %Bar.generic: %Bar.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar, @Bar(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc8_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc8_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc9: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc9: %C = bind_symbolic_name a, 0 [symbolic = %a.loc8_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc11_11.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc11_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc12: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc12: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc12: %C = bind_symbolic_name a, 0 [symbolic = %a.loc11_11.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo(%a.loc8_11.1: %C) {
// CHECK:STDOUT:   %a.loc8_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc8_11.2 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(%a.loc11_11.1: %C) {
// CHECK:STDOUT:   %a.loc11_11.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc11_11.2 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc8_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a.loc11_11.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = generic_class_type @Foo [concrete]
// CHECK:STDOUT:   %Foo.generic: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Foo: type = class_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc7_17.1: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_17.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc8: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_17.2 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo(%a.loc7_17.1: %C) {
// CHECK:STDOUT:   %a.loc7_17.2: %C = bind_symbolic_name a, 0 [symbolic = %a.loc7_17.2 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.loc7_17.2 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.f97b44.1: %C = import_ref Main//two_file, loc7_11, loaded [symbolic = @Foo.%a.1 (constants.%a)]
// CHECK:STDOUT:   %Main.import_ref.f97b44.2: %C = import_ref Main//two_file, loc8_11, loaded [symbolic = @Bar.%a.1 (constants.%a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Foo.decl: %Foo.type = class_decl @Foo [concrete = constants.%Foo.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc5: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = class_decl @Bar [concrete = constants.%Bar.generic] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc6: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Foo(imports.%Main.import_ref.f97b44.1: %C) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Bar(imports.%Main.import_ref.f97b44.2: %C) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Bar
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
