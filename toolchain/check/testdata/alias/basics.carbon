// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/basics.carbon

// --- alias.carbon

library "[[@TEST_NAME]]";

class C {};

//@dump-sem-ir-begin
alias c = C;
//@dump-sem-ir-end

let l: c = {};

// --- alias_of_alias.carbon

library "[[@TEST_NAME]]";

class C {}

//@dump-sem-ir-begin
alias a = C;
alias b = a;
alias c = b;
let d: c = {};
//@dump-sem-ir-end

// --- fail_control_flow.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+8]]:11: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+4]]:11: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~~~~~~~
// CHECK:STDERR:
alias a = true or false;

// --- fail_name_conflict.carbon

library "[[@TEST_NAME]]";

class C {}

alias a = C;
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+7]]:5: error: duplicate name `a` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: var a: C = {};
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-4]]:7: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: alias a = C;
// CHECK:STDERR:       ^
// CHECK:STDERR:
var a: C = {};

var b: C = {};
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+7]]:7: error: duplicate name `b` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: alias b = C;
// CHECK:STDERR:       ^
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-4]]:5: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: var b: C = {};
// CHECK:STDERR:     ^
// CHECK:STDERR:
alias b = C;

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";

var a: () = ();
var b: ()* = &a;

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+4]]:11: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias c = *b;
// CHECK:STDERR:           ^~
// CHECK:STDERR:
alias c = *b;

// --- fail_params.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_params.carbon:[[@LINE+8]]:8: error: `alias` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR: alias A(T:! type) = T*;
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_params.carbon:[[@LINE+4]]:21: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias A(T:! type) = T*;
// CHECK:STDERR:                     ^~
// CHECK:STDERR:
alias A(T:! type) = T*;

// --- fail_modifiers.carbon

library "[[@TEST_NAME]]";

class Class {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+25]]:10: error: `base` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+22]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:15: error: `default` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:23: error: `final` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:                       ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `abstract` not allowed on `alias` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract base default final alias A = Class;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `impl` not allowed on `alias` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: impl alias B = Class;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
impl alias B = Class;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `extern` not allowed on `alias` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: extern alias C = Class;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
extern alias C = Class;

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %C.decl [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- alias_of_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [concrete = constants.%C]
// CHECK:STDOUT:   %b: type = bind_alias b, %a [concrete = constants.%C]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [concrete = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %b [concrete = constants.%C]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.c48 = binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_13.2: init %C = class_init (), %.loc10_13.1 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_13.3: ref %C = temporary %.loc10_13.1, %.loc10_13.2
// CHECK:STDOUT:   %.loc10_13.4: ref %C = converted @__global_init.%.loc10, %.loc10_13.3
// CHECK:STDOUT:   %.loc10_13.5: %C = bind_value %.loc10_13.4
// CHECK:STDOUT:   %d: %C = bind_name d, %.loc10_13.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
