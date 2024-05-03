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

fn ReturnIncomplete() -> Class;

fn CallTakeIncomplete(p: Class*) {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+10]]:3: ERROR: Forming value of incomplete type `Class`.
  // CHECK:STDERR:   TakeIncomplete(*p);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-92]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-11]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete(*p);

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+10]]:3: ERROR: Forming value of incomplete type `Class`.
  // CHECK:STDERR:   TakeIncomplete({});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-104]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-23]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete({});
}

fn CallReturnIncomplete() {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+9]]:3: ERROR: Function returns incomplete type `Class`.
  // CHECK:STDERR:   ReturnIncomplete();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-118]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-35]]:23: Return type declared here.
  // CHECK:STDERR: fn ReturnIncomplete() -> Class;
  // CHECK:STDERR:                       ^~~~~~~~
  ReturnIncomplete();
}

// CHECK:STDOUT: --- fail_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %CallClassFunction: type = fn_type @CallClassFunction [template]
// CHECK:STDOUT:   %struct.2: CallClassFunction = struct_value () [template]
// CHECK:STDOUT:   %ConvertFromStruct: type = fn_type @ConvertFromStruct [template]
// CHECK:STDOUT:   %struct.3: ConvertFromStruct = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: type = ptr_type Class [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.4: G = struct_value () [template]
// CHECK:STDOUT:   %MemberAccess: type = fn_type @MemberAccess [template]
// CHECK:STDOUT:   %struct.5: MemberAccess = struct_value () [template]
// CHECK:STDOUT:   %Copy: type = fn_type @Copy [template]
// CHECK:STDOUT:   %struct.6: Copy = struct_value () [template]
// CHECK:STDOUT:   %Let: type = fn_type @Let [template]
// CHECK:STDOUT:   %struct.7: Let = struct_value () [template]
// CHECK:STDOUT:   %TakeIncomplete: type = fn_type @TakeIncomplete [template]
// CHECK:STDOUT:   %struct.8: TakeIncomplete = struct_value () [template]
// CHECK:STDOUT:   %ReturnIncomplete: type = fn_type @ReturnIncomplete [template]
// CHECK:STDOUT:   %struct.9: ReturnIncomplete = struct_value () [template]
// CHECK:STDOUT:   %CallTakeIncomplete: type = fn_type @CallTakeIncomplete [template]
// CHECK:STDOUT:   %struct.10: CallTakeIncomplete = struct_value () [template]
// CHECK:STDOUT:   %CallReturnIncomplete: type = fn_type @CallReturnIncomplete [template]
// CHECK:STDOUT:   %struct.11: CallReturnIncomplete = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .CallClassFunction = %CallClassFunction.decl
// CHECK:STDOUT:     .global_var = %global_var
// CHECK:STDOUT:     .ConvertFromStruct = %ConvertFromStruct.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .MemberAccess = %MemberAccess.decl
// CHECK:STDOUT:     .Copy = %Copy.decl
// CHECK:STDOUT:     .Let = %Let.decl
// CHECK:STDOUT:     .TakeIncomplete = %TakeIncomplete.decl
// CHECK:STDOUT:     .ReturnIncomplete = %ReturnIncomplete.decl
// CHECK:STDOUT:     .CallTakeIncomplete = %CallTakeIncomplete.decl
// CHECK:STDOUT:     .CallReturnIncomplete = %CallReturnIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %CallClassFunction.decl: CallClassFunction = fn_decl @CallClassFunction [template = constants.%struct.2] {}
// CHECK:STDOUT:   %Class.ref.loc36: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %global_var.var: ref <error> = var global_var
// CHECK:STDOUT:   %global_var: ref <error> = bind_name global_var, %global_var.var
// CHECK:STDOUT:   %ConvertFromStruct.decl: ConvertFromStruct = fn_decl @ConvertFromStruct [template = constants.%struct.3] {
// CHECK:STDOUT:     %Class.ref.loc45: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @ConvertFromStruct.%return: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.4] {
// CHECK:STDOUT:     %Class.ref.loc47: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc47: type = ptr_type Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc47_6.1: Class* = param p
// CHECK:STDOUT:     @G.%p: Class* = bind_name p, %p.loc47_6.1
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MemberAccess.decl: MemberAccess = fn_decl @MemberAccess [template = constants.%struct.5] {
// CHECK:STDOUT:     %Class.ref.loc58: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc58: type = ptr_type Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc58_17.1: Class* = param p
// CHECK:STDOUT:     @MemberAccess.%p: Class* = bind_name p, %p.loc58_17.1
// CHECK:STDOUT:     @MemberAccess.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.decl: Copy = fn_decl @Copy [template = constants.%struct.6] {
// CHECK:STDOUT:     %Class.ref.loc76_12: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc76: type = ptr_type Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc76_9.1: Class* = param p
// CHECK:STDOUT:     @Copy.%p: Class* = bind_name p, %p.loc76_9.1
// CHECK:STDOUT:     %Class.ref.loc76_23: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @Copy.%return: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let.decl: Let = fn_decl @Let [template = constants.%struct.7] {
// CHECK:STDOUT:     %Class.ref.loc80: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc80: type = ptr_type Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc80_8.1: Class* = param p
// CHECK:STDOUT:     @Let.%p: Class* = bind_name p, %p.loc80_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeIncomplete.decl: TakeIncomplete = fn_decl @TakeIncomplete [template = constants.%struct.8] {
// CHECK:STDOUT:     %Class.ref.loc91: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc91_19.1: Class = param c
// CHECK:STDOUT:     @TakeIncomplete.%c: Class = bind_name c, %c.loc91_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnIncomplete.decl: ReturnIncomplete = fn_decl @ReturnIncomplete [template = constants.%struct.9] {
// CHECK:STDOUT:     %Class.ref.loc93: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @ReturnIncomplete.%return: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTakeIncomplete.decl: CallTakeIncomplete = fn_decl @CallTakeIncomplete [template = constants.%struct.10] {
// CHECK:STDOUT:     %Class.ref.loc95: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc95: type = ptr_type Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc95_23.1: Class* = param p
// CHECK:STDOUT:     @CallTakeIncomplete.%p: Class* = bind_name p, %p.loc95_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallReturnIncomplete.decl: CallReturnIncomplete = fn_decl @CallReturnIncomplete [template = constants.%struct.11] {}
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
// CHECK:STDOUT: fn @ConvertFromStruct() -> Class {
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
// CHECK:STDOUT: fn @Copy(%p: Class*) -> Class {
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
// CHECK:STDOUT: fn @ReturnIncomplete() -> Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTakeIncomplete(%p: Class*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeIncomplete.ref.loc106: TakeIncomplete = name_ref TakeIncomplete, file.%TakeIncomplete.decl [template = constants.%struct.8]
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc106: ref Class = deref %p.ref
// CHECK:STDOUT:   %TakeIncomplete.call.loc106: init () = call %TakeIncomplete.ref.loc106(<invalid>) [template = <error>]
// CHECK:STDOUT:   %TakeIncomplete.ref.loc118: TakeIncomplete = name_ref TakeIncomplete, file.%TakeIncomplete.decl [template = constants.%struct.8]
// CHECK:STDOUT:   %.loc118: {} = struct_literal ()
// CHECK:STDOUT:   %TakeIncomplete.call.loc118: init () = call %TakeIncomplete.ref.loc118(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallReturnIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnIncomplete.ref: ReturnIncomplete = name_ref ReturnIncomplete, file.%ReturnIncomplete.decl [template = constants.%struct.9]
// CHECK:STDOUT:   %ReturnIncomplete.call: init <error> = call %ReturnIncomplete.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
