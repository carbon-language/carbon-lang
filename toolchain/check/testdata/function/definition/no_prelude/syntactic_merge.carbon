// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/no_prelude/syntactic_merge.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/no_prelude/syntactic_merge.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo(a: C);
fn Foo(a: C) {}

fn Bar(a: D);
fn Bar(a: D) {}

// --- spacing.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo [ ] ( a : C );
fn Foo[](a: C) {}

// --- fail_parens.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo(a: C);
// CHECK:STDERR: fail_parens.carbon:[[@LINE+7]]:11: error: redeclaration syntax differs here
// CHECK:STDERR: fn Foo(a: (C)) {}
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_parens.carbon:[[@LINE-4]]:11: note: comparing with previous declaration here
// CHECK:STDERR: fn Foo(a: C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
fn Foo(a: (C)) {}

// --- todo_fail_raw_identifier.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo(a: C);
fn Foo(a: r#C) {}

// --- two_file.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo(a: C);
fn Bar(a: D);

// --- two_file.impl.carbon

impl library "[[@TEST_NAME]]";

fn Foo(a: C) {}
fn Bar(a: D) {}

// --- fail_name_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo(a: C);
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE+7]]:8: error: redeclaration differs at parameter 1
// CHECK:STDERR: fn Foo(b: D) {}
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-4]]:8: note: previous declaration's corresponding parameter here
// CHECK:STDERR: fn Foo(a: C);
// CHECK:STDERR:        ^
// CHECK:STDERR:
fn Foo(b: D) {}

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo(a: C);
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:11: error: redeclaration syntax differs here
// CHECK:STDERR: fn Foo(a: D) {}
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE-4]]:11: note: comparing with previous declaration here
// CHECK:STDERR: fn Foo(a: C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
fn Foo(a: D) {}

// --- fail_deduced_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

fn Foo[a:! C]();
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE+7]]:12: error: redeclaration syntax differs here
// CHECK:STDERR: fn Foo[a:! D]() {}
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE-4]]:12: note: comparing with previous declaration here
// CHECK:STDERR: fn Foo[a:! C]();
// CHECK:STDERR:            ^
// CHECK:STDERR:
fn Foo[a:! D]() {}

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

fn Foo(a: D) {}

// --- fail_repeat_const.carbon

library "[[@TEST_NAME]]";

class C {}

fn Foo(a: const C);
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+10]]:11: warning: `const` applied repeatedly to the same type has no additional effect
// CHECK:STDERR: fn Foo(a: const (const C)) {}
// CHECK:STDERR:           ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+6]]:17: error: redeclaration syntax differs here
// CHECK:STDERR: fn Foo(a: const (const C)) {}
// CHECK:STDERR:                 ^
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE-8]]:17: note: comparing with previous declaration here
// CHECK:STDERR: fn Foo(a: const C);
// CHECK:STDERR:                 ^
fn Foo(a: const (const C)) {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [template]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [template]
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
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = param a, runtime_param0
// CHECK:STDOUT:     %a.loc7: %C = bind_name a, %a.param.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc8: %C = param a, runtime_param0
// CHECK:STDOUT:     %a.loc8: %C = bind_name a, %a.param.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type = fn_decl @Bar [template = constants.%Bar] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc10: %C = param a, runtime_param0
// CHECK:STDOUT:     %a.loc10: %C = bind_name a, %a.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type = fn_decl @Bar [template = constants.%Bar] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc11: %C = param a, runtime_param0
// CHECK:STDOUT:     %a.loc11: %C = bind_name a, %a.param.loc11
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
// CHECK:STDOUT: fn @Foo(%a.loc8: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar(%a.loc11: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc6: %C = param a, runtime_param0
// CHECK:STDOUT:     %a.loc6: %C = bind_name a, %a.param.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = param a, runtime_param0
// CHECK:STDOUT:     %a.loc7: %C = bind_name a, %a.param.loc7
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
// CHECK:STDOUT: fn @Foo[](%a.loc7: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%a: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc6: %C = param a, runtime_param0
// CHECK:STDOUT:     %a.loc6: %C = bind_name a, %a.param.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param.loc7: %C = param a, runtime_param0
// CHECK:STDOUT:     %a.loc7: %C = bind_name a, %a.param.loc7
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
// CHECK:STDOUT: fn @Foo(%a.loc7: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [template]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [template]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [template = constants.%Bar] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar(%a: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %Bar.type: type = fn_type @Bar [template]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//two_file, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//two_file, inst+7, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3: %Foo.type = import_ref Main//two_file, inst+12, loaded [template = constants.%Foo]
// CHECK:STDOUT:   %import_ref.4: %Bar.type = import_ref Main//two_file, inst+20, loaded [template = constants.%Bar]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = fn_decl @Bar [template = constants.%Bar] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.2 [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar(%a: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.1 [template]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %b.param: %C = param b, runtime_param0
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
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
// CHECK:STDOUT: fn @Foo(%a: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%b: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.1 [template]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%a: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduced_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.1 [template]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = symbolic_binding_pattern a, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %a.patt: %C = symbolic_binding_pattern a, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc15: %C = bind_symbolic_name a, 0, %a.param [symbolic = %a.1 (constants.%a)]
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
// CHECK:STDOUT: generic fn @Foo(%a.loc7: %C) {
// CHECK:STDOUT:   %a.1: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%a.loc7: %C]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%a.loc15: %C) {
// CHECK:STDOUT:   %a.1: %C = bind_symbolic_name a, 0 [symbolic = %a.1 (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%a.loc15: %C]() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_alias_in_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = fn_decl @Foo [template = constants.%Foo] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var.loc7: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type = fn_decl @Foo [template = constants.%Foo] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
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
// CHECK:STDOUT: fn @Foo() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_25.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc8_25.2: init %C = class_init (), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc8_26: init %C = converted %.loc8_25.1, %.loc8_25.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc8_26 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//alias_two_file, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: %Foo.type = import_ref Main//alias_two_file, inst+10, loaded [template = constants.%Foo]
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
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeat_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = const_type %C [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.5: %.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %a.patt: %.3 = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc6: type = const_type %C [template = constants.%.3]
// CHECK:STDOUT:     %a.param: %.3 = param a, runtime_param0
// CHECK:STDOUT:     %a: %.3 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.5] {
// CHECK:STDOUT:     %a.patt: %.3 = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc17_18: type = const_type %C [template = constants.%.3]
// CHECK:STDOUT:     %.loc17_11: type = const_type %.3 [template = constants.%.3]
// CHECK:STDOUT:     %a.param: %.3 = param a, runtime_param0
// CHECK:STDOUT:     %a: %.3 = bind_name a, %a.param
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
// CHECK:STDOUT: fn @Foo(%a: %.3);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%a: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
