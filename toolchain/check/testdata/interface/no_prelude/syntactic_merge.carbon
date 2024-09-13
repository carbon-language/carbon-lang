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
// CHECK:STDERR: fail_parens.carbon:[[@LINE+7]]:19: ERROR: Redeclaration syntax differs here.
// CHECK:STDERR: interface Foo(a:! (C)) {}
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_parens.carbon:[[@LINE-4]]:19: Comparing with previous declaration here.
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

// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE+10]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: interface Foo(a:! C) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE-5]]:6: In import.
// CHECK:STDERR: impl library "two_file";
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: two_file.carbon:7:1: Name is previously declared here.
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Foo(a:! C) {}
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE+10]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: interface Bar(a:! D) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_two_file.impl.carbon:[[@LINE-16]]:6: In import.
// CHECK:STDERR: impl library "two_file";
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: two_file.carbon:8:1: Name is previously declared here.
// CHECK:STDERR: interface Bar(a:! D);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Bar(a:! D) {}

// --- fail_name_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a:! C);
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE+7]]:15: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: interface Foo(b:! D) {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_name_mismatch.carbon:[[@LINE-4]]:15: Previous declaration's corresponding parameter here.
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR:               ^
// CHECK:STDERR:
interface Foo(b:! D) {}

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo(a:! C);
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:19: ERROR: Redeclaration syntax differs here.
// CHECK:STDERR: interface Foo(a:! D) {}
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE-4]]:19: Comparing with previous declaration here.
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR:                   ^
// CHECK:STDERR:
interface Foo(a:! D) {}

// --- fail_deduced_alias.carbon

library "[[@TEST_NAME]]";

class C {}
alias D = C;

interface Foo[a:! C]();
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE+7]]:19: ERROR: Redeclaration syntax differs here.
// CHECK:STDERR: interface Foo[a:! D]() {}
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_deduced_alias.carbon:[[@LINE-4]]:19: Comparing with previous declaration here.
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
// CHECK:STDERR: fail_alias_two_file.impl.carbon:[[@LINE+10]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: interface Foo(a:! D) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_alias_two_file.impl.carbon:[[@LINE-10]]:6: In import.
// CHECK:STDERR: impl library "alias_two_file";
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: alias_two_file.carbon:6:1: Name is previously declared here.
// CHECK:STDERR: interface Foo(a:! C);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Foo(a:! D) {}

// --- fail_repeat_const.carbon

library "[[@TEST_NAME]]";

class C {}

interface Foo(a:! const C);
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+10]]:19: WARNING: `const` applied repeatedly to the same type has no additional effect.
// CHECK:STDERR: interface Foo(a:! const (const C)) {}
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE+6]]:25: ERROR: Redeclaration syntax differs here.
// CHECK:STDERR: interface Foo(a:! const (const C)) {}
// CHECK:STDERR:                         ^
// CHECK:STDERR: fail_repeat_const.carbon:[[@LINE-8]]:25: Comparing with previous declaration here.
// CHECK:STDERR: interface Foo(a:! const C);
// CHECK:STDERR:                         ^
interface Foo(a:! const (const C)) {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = interface_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT:   %Self.1: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Bar.type: type = generic_interface_type @Bar [template]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @Bar, @Bar(%a) [symbolic]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl.loc7
// CHECK:STDOUT:     .Bar = %Bar.decl.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref.loc5: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.2: %C = bind_symbolic_name a 0, %a.loc7_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc8: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc8_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc8_15.2: %C = bind_symbolic_name a 0, %a.loc8_15.1 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc10: %Bar.type = interface_decl @Bar [template = constants.%Bar] {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:     %a.loc10_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc10_15.2: %C = bind_symbolic_name a 0, %a.loc10_15.1 [symbolic = @Bar.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl.loc11: %Bar.type = interface_decl @Bar [template = constants.%Bar] {
// CHECK:STDOUT:     %D.ref.loc11: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:     %a.loc11_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc11_15.2: %C = bind_symbolic_name a 0, %a.loc11_15.1 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc7_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Foo, @Foo(%a) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(file.%a.loc10_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Bar, @Bar(%a) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Bar.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(@Foo.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(@Bar.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- spacing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = interface_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc6_21.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_21.2: %C = bind_symbolic_name a 0, %a.loc6_21.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc7_17.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_17.2: %C = bind_symbolic_name a 0, %a.loc7_17.1 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc6_21.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Foo, @Foo(%a) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(@Foo.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @.1, @.1(%a) [symbolic]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc6_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_15.2: %C = bind_symbolic_name a 0, %a.loc6_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = interface_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc14_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc14_15.2: %C = bind_symbolic_name a 0, %a.loc14_15.1 [symbolic = @.1.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc6_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(file.%a.loc14_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.1, @.1(%a) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(@.1.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = interface_type @Foo, @Foo(%a) [symbolic]
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %Foo.decl.loc6: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc6_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_15.2: %C = bind_symbolic_name a 0, %a.loc6_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl.loc7: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.2: %C = bind_symbolic_name a 0, %a.loc7_15.1 [symbolic = constants.%a]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc6_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @Foo, @Foo(%a) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Foo.%.1 (%.3) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(@Foo.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %Bar.type: type = generic_interface_type @Bar [template]
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref.loc5: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.2: %C = bind_symbolic_name a 0, %a.loc7_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: %Bar.type = interface_decl @Bar [template = constants.%Bar] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:     %a.loc8_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc8_15.2: %C = bind_symbolic_name a 0, %a.loc8_15.1 [symbolic = @Bar.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc7_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(file.%a.loc8_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @.1, @.1(%a) [symbolic]
// CHECK:STDOUT:   %Self.1: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Bar.type: type = generic_interface_type @Bar [template]
// CHECK:STDOUT:   %Bar: %Bar.type = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = generic_interface_type @.2 [template]
// CHECK:STDOUT:   %.5: %.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = interface_type @.2, @.2(%a) [symbolic]
// CHECK:STDOUT:   %Self.2: %.6 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//two_file, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//two_file, inst+5, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3: %Foo.type = import_ref Main//two_file, inst+10, loaded [template = constants.%Foo]
// CHECK:STDOUT:   %import_ref.4: %Bar.type = import_ref Main//two_file, inst+18, loaded [template = constants.%Bar]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//two_file, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .D = imports.%import_ref.2
// CHECK:STDOUT:     .Foo = imports.%import_ref.3
// CHECK:STDOUT:     .Bar = imports.%import_ref.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %.decl.loc14: %.type.1 = interface_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %a.loc14_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc14_15.2: %C = bind_symbolic_name a 0, %a.loc14_15.1 [symbolic = @.1.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc25: %.type.2 = interface_decl @.2 [template = constants.%.5] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.2 [template = constants.%C]
// CHECK:STDOUT:     %a.loc25_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc25_15.2: %C = bind_symbolic_name a 0, %a.loc25_15.1 [symbolic = @.2.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(constants.%a: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(file.%a.loc14_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.1, @.1(%a) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Bar(constants.%a: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.2(file.%a.loc25_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.2, @.2(%a) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %Self.2: %.6 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.2.%.1 (%.6) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(@.1.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Bar(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.2(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.2(@.2.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %b: %C = bind_symbolic_name b 0 [symbolic]
// CHECK:STDOUT:   %.type: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @.1, @.1(%b) [symbolic]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref.loc5: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.2: %C = bind_symbolic_name a 0, %a.loc7_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = interface_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:     %b.loc15_15.1: %C = param b, runtime_param<invalid>
// CHECK:STDOUT:     %b.loc15_15.2: %C = bind_symbolic_name b 0, %b.loc15_15.1 [symbolic = @.1.%b (constants.%b)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc7_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(file.%b.loc15_15.2: %C) {
// CHECK:STDOUT:   %b: %C = bind_symbolic_name b 0 [symbolic = %b (constants.%b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.1, @.1(%b) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%b) {
// CHECK:STDOUT:   %b => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(@.1.%b) {
// CHECK:STDOUT:   %b => constants.%b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @.1, @.1(%a) [symbolic]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref.loc5: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.2: %C = bind_symbolic_name a 0, %a.loc7_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = interface_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:     %a.loc15_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc15_15.2: %C = bind_symbolic_name a 0, %a.loc15_15.1 [symbolic = @.1.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc7_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(file.%a.loc15_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.1, @.1(%a) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(@.1.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduced_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @.1, @.1(%a) [symbolic]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref.loc5: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc7_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc7_15.2: %C = bind_symbolic_name a 0, %a.loc7_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = interface_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:     %a.loc15_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc15_15.2: %C = bind_symbolic_name a 0, %a.loc15_15.1 [symbolic = @.1.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc7_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(file.%a.loc15_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.1, @.1(%a) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(@.1.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_two_file.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc6_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_15.2: %C = bind_symbolic_name a 0, %a.loc6_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc6_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias_two_file.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.3: %.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @.1, @.1(%a) [symbolic]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//alias_two_file, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: %Foo.type = import_ref Main//alias_two_file, inst+8, loaded [template = constants.%Foo]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//alias_two_file, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .Foo = imports.%import_ref.2
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %.decl: %.type = interface_decl @.1 [template = constants.%.3] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:     %a.loc19_15.1: %C = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc19_15.2: %C = bind_symbolic_name a 0, %a.loc19_15.1 [symbolic = @.1.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(constants.%a: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(file.%a.loc19_15.2: %C) {
// CHECK:STDOUT:   %a: %C = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.1, @.1(%a) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(@.1.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeat_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = const_type %C [template]
// CHECK:STDOUT:   %a: %.2 = bind_symbolic_name a 0 [symbolic]
// CHECK:STDOUT:   %Foo.type: type = generic_interface_type @Foo [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.type: type = generic_interface_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = interface_type @.1, @.1(%a) [symbolic]
// CHECK:STDOUT:   %Self: %.5 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = interface_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc6: type = const_type %C [template = constants.%.2]
// CHECK:STDOUT:     %a.loc6_15.1: %.2 = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc6_15.2: %.2 = bind_symbolic_name a 0, %a.loc6_15.1 [symbolic = @Foo.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = interface_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %C.ref.loc17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc17_26: type = const_type %C [template = constants.%.2]
// CHECK:STDOUT:     %.loc17_19: type = const_type %.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.loc17_15.1: %.2 = param a, runtime_param<invalid>
// CHECK:STDOUT:     %a.loc17_15.2: %.2 = bind_symbolic_name a 0, %a.loc17_15.1 [symbolic = @.1.%a (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Foo(file.%a.loc6_15.2: %.2) {
// CHECK:STDOUT:   %a: %.2 = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @.1(file.%a.loc17_15.2: %.2) {
// CHECK:STDOUT:   %a: %.2 = bind_symbolic_name a 0 [symbolic = %a (constants.%a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @.1, @.1(%a) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %Self.2: %.5 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @.1.%.1 (%.5) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(@.1.%a) {
// CHECK:STDOUT:   %a => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
