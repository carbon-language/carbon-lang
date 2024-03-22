// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class;

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:4: ERROR: Cannot declare a member of incomplete class `Class`.
// CHECK:STDERR: fn Class.Function() {}
// CHECK:STDERR:    ^~~~~
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-5]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn Class.Function() {}

fn CallClassFunction() {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:3: ERROR: Member access into incomplete class `Class`.
  // CHECK:STDERR:   Class.Function();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-15]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  Class.Function();
}

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:17: ERROR: Variable has incomplete type `Class`.
// CHECK:STDERR: var global_var: Class;
// CHECK:STDERR:                 ^~~~~
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-25]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var global_var: Class;

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:24: ERROR: Function returns incomplete type `Class`.
// CHECK:STDERR: fn ConvertFromStruct() -> Class { return {}; }
// CHECK:STDERR:                        ^~~~~~~~
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-34]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn ConvertFromStruct() -> Class { return {}; }

fn G(p: Class*) -> i32 {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:10: ERROR: Member access into object of incomplete type `Class`.
  // CHECK:STDERR:   return p->n;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-44]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  return p->n;
}

fn MemberAccess(p: Class*) -> i32 {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:11: ERROR: Member access into object of incomplete type `Class`.
  // CHECK:STDERR:   return (*p).n;
  // CHECK:STDERR:           ^~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-55]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  return (*p).n;
}

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:20: ERROR: Function returns incomplete type `Class`.
// CHECK:STDERR: fn Copy(p: Class*) -> Class {
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-65]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn Copy(p: Class*) -> Class {
  return *p;
}

fn Let(p: Class*) {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:10: ERROR: `let` binding has incomplete type `Class`.
  // CHECK:STDERR:   let c: Class = *p;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-77]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  let c: Class = *p;
}

fn TakeIncomplete(c: Class);

// TODO: We should allow this, and only reject calls to the function.
//
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:23: ERROR: Function returns incomplete type `Class`.
// CHECK:STDERR: fn ReturnIncomplete() -> Class;
// CHECK:STDERR:                       ^~~~~~~~
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-91]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn ReturnIncomplete() -> Class;

fn CallTakeIncomplete(p: Class*) {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+10]]:3: ERROR: Forming value of incomplete type `Class`.
  // CHECK:STDERR:   TakeIncomplete(*p);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-101]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-20]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete(*p);

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+9]]:3: ERROR: Forming value of incomplete type `Class`.
  // CHECK:STDERR:   TakeIncomplete({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-113]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-32]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  TakeIncomplete({});
}

fn CallReturnIncomplete() {
  ReturnIncomplete();
}

// CHECK:STDOUT: --- fail_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = ptr_type Class [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CallClassFunction = %CallClassFunction
// CHECK:STDOUT:     .global_var = %global_var
// CHECK:STDOUT:     .ConvertFromStruct = %ConvertFromStruct
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:     .MemberAccess = %MemberAccess
// CHECK:STDOUT:     .Copy = %Copy
// CHECK:STDOUT:     .Let = %Let
// CHECK:STDOUT:     .TakeIncomplete = %TakeIncomplete
// CHECK:STDOUT:     .ReturnIncomplete = %ReturnIncomplete
// CHECK:STDOUT:     .CallTakeIncomplete = %CallTakeIncomplete
// CHECK:STDOUT:     .CallReturnIncomplete = %CallReturnIncomplete
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %.loc16: <function> = fn_decl @.1 [template] {}
// CHECK:STDOUT:   %CallClassFunction: <function> = fn_decl @CallClassFunction [template] {}
// CHECK:STDOUT:   %Class.ref.loc36: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %global_var.var: ref <error> = var global_var
// CHECK:STDOUT:   %global_var: ref <error> = bind_name global_var, %global_var.var
// CHECK:STDOUT:   %ConvertFromStruct: <function> = fn_decl @ConvertFromStruct [template] {
// CHECK:STDOUT:     %Class.ref.loc45: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %return.var.loc45: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %Class.ref.loc47: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc47: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc47_6.1: Class* = param p
// CHECK:STDOUT:     @G.%p: Class* = bind_name p, %p.loc47_6.1
// CHECK:STDOUT:     %return.var.loc47: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MemberAccess: <function> = fn_decl @MemberAccess [template] {
// CHECK:STDOUT:     %Class.ref.loc58: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc58: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc58_17.1: Class* = param p
// CHECK:STDOUT:     @MemberAccess.%p: Class* = bind_name p, %p.loc58_17.1
// CHECK:STDOUT:     %return.var.loc58: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy: <function> = fn_decl @Copy [template] {
// CHECK:STDOUT:     %Class.ref.loc76_12: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc76: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc76_9.1: Class* = param p
// CHECK:STDOUT:     @Copy.%p: Class* = bind_name p, %p.loc76_9.1
// CHECK:STDOUT:     %Class.ref.loc76_23: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %return.var.loc76: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let: <function> = fn_decl @Let [template] {
// CHECK:STDOUT:     %Class.ref.loc80: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc80: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc80_8.1: Class* = param p
// CHECK:STDOUT:     @Let.%p: Class* = bind_name p, %p.loc80_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeIncomplete: <function> = fn_decl @TakeIncomplete [template] {
// CHECK:STDOUT:     %Class.ref.loc91: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc91_19.1: Class = param c
// CHECK:STDOUT:     @TakeIncomplete.%c: Class = bind_name c, %c.loc91_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnIncomplete: <function> = fn_decl @ReturnIncomplete [template] {
// CHECK:STDOUT:     %Class.ref.loc102: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %return.var.loc102: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTakeIncomplete: <function> = fn_decl @CallTakeIncomplete [template] {
// CHECK:STDOUT:     %Class.ref.loc104: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc104: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc104_23.1: Class* = param p
// CHECK:STDOUT:     @CallTakeIncomplete.%p: Class* = bind_name p, %p.loc104_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallReturnIncomplete: <function> = fn_decl @CallReturnIncomplete [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallClassFunction() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %Function.ref: <error> = name_ref Function, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertFromStruct() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc45: {} = struct_literal ()
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc55: ref Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberAccess(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc66: ref Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Copy(%p: Class*) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc77: ref Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let(%p: Class*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc88: ref Class = deref %p.ref
// CHECK:STDOUT:   %c: <error> = bind_name c, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeIncomplete(%c: Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnIncomplete() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTakeIncomplete(%p: Class*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeIncomplete.ref.loc115: <function> = name_ref TakeIncomplete, file.%TakeIncomplete [template = file.%TakeIncomplete]
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc115_18: ref Class = deref %p.ref
// CHECK:STDOUT:   %.loc115_17: init () = call %TakeIncomplete.ref.loc115(<invalid>) [template = <error>]
// CHECK:STDOUT:   %TakeIncomplete.ref.loc126: <function> = name_ref TakeIncomplete, file.%TakeIncomplete [template = file.%TakeIncomplete]
// CHECK:STDOUT:   %.loc126_19: {} = struct_literal ()
// CHECK:STDOUT:   %.loc126_17: init () = call %TakeIncomplete.ref.loc126(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallReturnIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnIncomplete.ref: <function> = name_ref ReturnIncomplete, file.%ReturnIncomplete [template = file.%ReturnIncomplete]
// CHECK:STDOUT:   %.loc130: init <error> = call %ReturnIncomplete.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
