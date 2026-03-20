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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/syntactic_merge.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/syntactic_merge.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo(a: C);
fn Foo(unused a: C) {}

fn Bar(a: D);
fn Bar(unused a: D) {}

// --- spacing.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo [ ] ( a : C );
fn Foo[](unused a: C) {}

// --- fail_parens.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo(a: C);
// CHECK:STDERR: fail_parens.carbon:[[@LINE+7]]:18: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Foo(unused a: (C)) {}
// CHECK:STDERR:                  ^
// CHECK:STDERR: fail_parens.carbon:[[@LINE-4]]:11: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: fn Foo(a: C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
fn Foo(unused a: (C)) {}

// --- fail_only_implicit_params.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_only_implicit_params.carbon:[[@LINE+8]]:8: error: implicit parameters of functions must be constant or `self` [ImplictParamMustBeConstant]
// CHECK:STDERR: fn Foo[a: C];
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_only_implicit_params.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn Foo[a: C];
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn Foo[a: C];
// CHECK:STDERR: fail_only_implicit_params.carbon:[[@LINE+8]]:8: error: implicit parameters of functions must be constant or `self` [ImplictParamMustBeConstant]
// CHECK:STDERR: fn Foo[a: (C)] {}
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_only_implicit_params.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn Foo[a: (C)] {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Foo[a: (C)] {}

// --- todo_fail_raw_identifier.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo(a: C);
fn Foo(unused a: r#C) {}

// --- two_file.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo(a: C);
fn Bar(a: D);

// --- two_file.impl.carbon

impl library "[[@TEST_NAME]]";

fn Foo(unused a: C) {}
fn Bar(unused a: D) {}

// --- fail_name_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo(a: C);
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE+7]]:15: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fn Foo(unused b: D) {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-4]]:8: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: fn Foo(a: C);
// CHECK:STDERR:        ^
// CHECK:STDERR:
fn Foo(unused b: D) {}

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo(a: C);
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:18: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Foo(unused a: D) {}
// CHECK:STDERR:                  ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE-4]]:11: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: fn Foo(a: C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
fn Foo(unused a: D) {}

// --- fail_deduced_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo[a:! C]();
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE+7]]:19: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Foo[unused a:! D]() {}
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE-4]]:12: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: fn Foo[a:! C]();
// CHECK:STDERR:            ^
// CHECK:STDERR:
fn Foo[unused a:! D]() {}

// --- todo_fail_alias_in_return.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo() -> C;
fn Foo() -> D { return {}; }

// --- alias_two_file.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo(a: C);

// --- todo_fail_alias_two_file.impl.carbon

impl library "[[@TEST_NAME]]";

alias D = C;

fn Foo(unused a: D) {}

// --- fail_repeat_const.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo(a: const C);
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+11]]:18: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn Foo(unused a: const (const C)) {}
// CHECK:STDERR:                  ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+7]]:24: error: redeclaration syntax differs here [RedeclParamSyntaxDiffers]
// CHECK:STDERR: fn Foo(unused a: const (const C)) {}
// CHECK:STDERR:                        ^
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE-8]]:17: note: comparing with previous declaration here [RedeclParamSyntaxPrevious]
// CHECK:STDERR: fn Foo(a: const C);
// CHECK:STDERR:                 ^
// CHECK:STDERR:
fn Foo(unused a: const (const C)) {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [concrete]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc20
// CHECK:STDOUT:     .Bar = %Bar.decl.loc38
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc20: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param.loc17: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc16: %C = value_binding a, %a.param.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc29: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param.loc26: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc27: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc25: %C = value_binding a, %a.param.loc26
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc38: %Bar.type = fn_decl @Bar [concrete = constants.%Bar] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param.loc35: %C = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc36: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc34: %C = value_binding a, %a.param.loc35
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc47: %Bar.type = fn_decl @Bar [concrete = constants.%Bar] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param.loc44: %C = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc45: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc43: %C = value_binding a, %a.param.loc44
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
// CHECK:STDOUT: fn @Foo(%a.param.loc26: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar(%a.param.loc44: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc17: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param.loc14: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc13: %C = value_binding a, %a.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc28: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param.loc25: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc24: %C = value_binding a, %a.param.loc25
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
// CHECK:STDOUT: fn @Foo(%a.param.loc25: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.1: type = fn_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.1: %Foo.type.47530a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.2: type = fn_type @Foo.loc26 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.2: %Foo.type.47530a.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.47530a.1 = fn_decl @Foo.loc15 [concrete = constants.%Foo.9ec12f.1] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc26: %Foo.type.47530a.2 = fn_decl @Foo.loc26 [concrete = constants.%Foo.9ec12f.2] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
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
// CHECK:STDOUT: fn @Foo.loc15(%a.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo.loc26(%a.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_only_implicit_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.1: type = fn_type @Foo.loc15 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.1: %Foo.type.47530a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.2: type = fn_type @Foo.loc25 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.2: %Foo.type.47530a.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type.47530a.1 = fn_decl @Foo.loc15 [concrete = constants.%Foo.9ec12f.1] {} {}
// CHECK:STDOUT:   %Foo.decl.loc25: %Foo.type.47530a.2 = fn_decl @Foo.loc25 [concrete = constants.%Foo.9ec12f.2] {} {}
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
// CHECK:STDOUT: fn @Foo.loc15();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo.loc25() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc15: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param.loc12: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc11: %C = value_binding a, %a.param.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc24: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param.loc21: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a.loc20: %C = value_binding a, %a.param.loc21
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
// CHECK:STDOUT: fn @Foo(%a.param.loc21: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [concrete]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [concrete = constants.%Bar] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar(%a.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [concrete]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//two_file, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//two_file, loc7, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [concrete = constants.%Bar] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a.param: %C) [from "two_file.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar(%a.param: %C) [from "two_file.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.1: type = fn_type @Foo.loc20 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.1: %Foo.type.47530a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.2: type = fn_type @Foo.loc29 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.2: %Foo.type.47530a.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Foo.decl.loc20: %Foo.type.47530a.1 = fn_decl @Foo.loc20 [concrete = constants.%Foo.9ec12f.1] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc29: %Foo.type.47530a.2 = fn_decl @Foo.loc29 [concrete = constants.%Foo.9ec12f.2] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %C = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %b: %C = value_binding b, %b.param
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
// CHECK:STDOUT: fn @Foo.loc20(%a.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo.loc29(%b.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.1: type = fn_type @Foo.loc20 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.1: %Foo.type.47530a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.2: type = fn_type @Foo.loc29 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.2: %Foo.type.47530a.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Foo.decl.loc20: %Foo.type.47530a.1 = fn_decl @Foo.loc20 [concrete = constants.%Foo.9ec12f.1] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc29: %Foo.type.47530a.2 = fn_decl @Foo.loc29 [concrete = constants.%Foo.9ec12f.2] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
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
// CHECK:STDOUT: fn @Foo.loc20(%a.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo.loc29(%a.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
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
// CHECK:STDOUT:   %a.9daf47.1: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.47530a.1: type = fn_type @Foo.loc22 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.1: %Foo.type.47530a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %a.9daf47.2: %C = symbolic_binding a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type.47530a.2: type = fn_type @Foo.loc33 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.2: %Foo.type.47530a.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Foo.decl.loc22: %Foo.type.47530a.1 = fn_decl @Foo.loc22 [concrete = constants.%Foo.9ec12f.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc16.2: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a.9daf47.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc33: %Foo.type.47530a.2 = fn_decl @Foo.loc33 [concrete = constants.%Foo.9ec12f.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type = symbolic_binding_pattern a, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc29: type = splice_block %D.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc27.3: %C = symbolic_binding a, 0 [symbolic = %a.loc27.2 (constants.%a.9daf47.2)]
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
// CHECK:STDOUT: generic fn @Foo.loc22(%a.loc16.2: %C) {
// CHECK:STDOUT:   %a.loc16.1: %C = symbolic_binding a, 0 [symbolic = %a.loc16.1 (constants.%a.9daf47.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Foo.loc33(%a.loc27.3: %C) {
// CHECK:STDOUT:   %a.loc27.1: %C = symbolic_binding a, 0 [symbolic = %a.loc27.1 (constants.%a.9daf47.1)]
// CHECK:STDOUT:   %a.loc27.2: %C = symbolic_binding a, 0 [symbolic = %a.loc27.2 (constants.%a.9daf47.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc22(constants.%a.9daf47.1) {
// CHECK:STDOUT:   %a.loc16.1 => constants.%a.9daf47.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo.loc33(constants.%a.9daf47.2) {
// CHECK:STDOUT:   %a.loc27.1 => constants.%a.9daf47.2
// CHECK:STDOUT:   %a.loc27.2 => constants.%a.9daf47.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_alias_in_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.a69: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc19: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %C.ref [concrete = constants.%.a69]
// CHECK:STDOUT:     %return.param.loc18: ref %C = out_param call_param0
// CHECK:STDOUT:     %return.loc18: ref %C = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc26: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %.loc25: Core.Form = init_form %D.ref [concrete = constants.%.a69]
// CHECK:STDOUT:     %return.param.loc25: ref %C = out_param call_param0
// CHECK:STDOUT:     %return.loc25: ref %C = return_slot %return.param.loc25
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
// CHECK:STDOUT: fn @Foo() -> out %return.param.loc25: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc29.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc29.2: init %C to %return.param.loc25 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc30: init %C = converted %.loc29.1, %.loc29.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc30 to %return.param.loc25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//alias_two_file, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//alias_two_file, loc7, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//alias_two_file, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = alias_binding D, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %C = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [concrete = constants.%C]
// CHECK:STDOUT:     %a: %C = value_binding a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a.param: %C) [from "alias_two_file.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeat_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.1: type = fn_type @Foo.loc16 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.1: %Foo.type.47530a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Foo.type.47530a.2: type = fn_type @Foo.loc29 [concrete]
// CHECK:STDOUT:   %Foo.9ec12f.2: %Foo.type.47530a.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc16: %Foo.type.47530a.1 = fn_decl @Foo.loc16 [concrete = constants.%Foo.9ec12f.1] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %const = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %const [concrete = constants.%const] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %const = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc29: %Foo.type.47530a.2 = fn_decl @Foo.loc29 [concrete = constants.%Foo.9ec12f.2] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %const = value_param call_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %const.loc23 [concrete = constants.%const] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc25: type = const_type %C.ref [concrete = constants.%const]
// CHECK:STDOUT:       %const.loc23: type = const_type %const.loc25 [concrete = constants.%const]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %const = value_binding a, %a.param
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
// CHECK:STDOUT: fn @Foo.loc16(%a.param: %const);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo.loc29(%a.param: %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
