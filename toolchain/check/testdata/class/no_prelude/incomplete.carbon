// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/incomplete.carbon

// --- forward_decl.carbon

//@dump-sem-ir-begin
class Class;

fn F(p: Class*) -> Class* { return p; }
//@dump-sem-ir-end

// --- fail_forward_decl.carbon

library "[[@TEST_NAME]]";

class Class;

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:4: error: cannot declare a member of incomplete class `Class` [QualifiedDeclInIncompleteClassScope]
// CHECK:STDERR: fn Class.Function() {}
// CHECK:STDERR:    ^~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn Class.Function() {}

fn CallClassFunction() {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:3: error: member access into incomplete class `Class` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR:   Class.Function();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-15]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  Class.Function();
}

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:17: error: binding pattern has incomplete type `Class` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var global_var: Class;
// CHECK:STDERR:                 ^~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-25]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var global_var: Class;

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:24: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn ConvertFromStruct() -> Class { return {}; }
// CHECK:STDERR:                        ^~~~~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-34]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn ConvertFromStruct() -> Class { return {}; }

fn G(p: Class*) -> () {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:10: error: member access into object of incomplete type `Class` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   return p->n;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-44]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  return p->n;
}

fn MemberAccess(p: Class*) -> () {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:11: error: member access into object of incomplete type `Class` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   return (*p).n;
  // CHECK:STDERR:           ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-55]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  return (*p).n;
}

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:20: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn Copy(p: Class*) -> Class {
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-65]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn Copy(p: Class*) -> Class {
  return *p;
}

fn Let(p: Class*) {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:10: error: binding pattern has incomplete type `Class` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   let c: Class = *p;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-77]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  let c: Class = *p;
}

fn TakeIncomplete(c: Class);

fn ReturnIncomplete() -> Class;

fn CallTakeIncomplete(p: Class*) {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:18: error: forming value of incomplete type `Class` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   TakeIncomplete(*p);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-92]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-11]]:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete(*p);

  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:18: error: forming value of incomplete type `Class` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   TakeIncomplete({});
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-104]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-23]]:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete({});
}

fn CallReturnIncomplete() {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:3: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnIncomplete();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-118]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-35]]:23: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnIncomplete() -> Class;
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR:
  ReturnIncomplete();
}

class IncompleteAddrSelf {
  fn F[addr self: Class*]();
}

fn CallIncompleteAddrSelf(p: Class*) {
  // TODO: Should this be valid?
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:3: error: invalid use of incomplete type `Class` [IncompleteTypeInConversion]
  // CHECK:STDERR:   p->(IncompleteAddrSelf.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-137]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-11]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->(IncompleteAddrSelf.F)();
}

// --- fail_in_definition.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE+7]]:10: error: field has incomplete type `C` [IncompleteTypeInFieldDecl]
  // CHECK:STDERR:   var c: C;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE-4]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~~~~~
  // CHECK:STDERR:
  var c: C;
}

// CHECK:STDOUT: --- forward_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.patt: %pattern_type = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc5_20: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc5_25: type = ptr_type %Class.ref.loc5_20 [concrete = constants.%ptr]
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %ptr.loc5_14 [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Class.ref.loc5_9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr.loc5_14: type = ptr_type %Class.ref.loc5_9 [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %ptr) -> %ptr {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
