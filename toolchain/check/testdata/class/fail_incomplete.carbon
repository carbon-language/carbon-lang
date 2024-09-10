// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_incomplete.carbon

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
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: %.type = struct_value () [template]
// CHECK:STDOUT:   %CallClassFunction.type: type = fn_type @CallClassFunction [template]
// CHECK:STDOUT:   %CallClassFunction: %CallClassFunction.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertFromStruct.type: type = fn_type @ConvertFromStruct [template]
// CHECK:STDOUT:   %ConvertFromStruct: %ConvertFromStruct.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: type = ptr_type %Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %MemberAccess.type: type = fn_type @MemberAccess [template]
// CHECK:STDOUT:   %MemberAccess: %MemberAccess.type = struct_value () [template]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [template]
// CHECK:STDOUT:   %Copy: %Copy.type = struct_value () [template]
// CHECK:STDOUT:   %Let.type: type = fn_type @Let [template]
// CHECK:STDOUT:   %Let: %Let.type = struct_value () [template]
// CHECK:STDOUT:   %TakeIncomplete.type: type = fn_type @TakeIncomplete [template]
// CHECK:STDOUT:   %TakeIncomplete: %TakeIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnIncomplete.type: type = fn_type @ReturnIncomplete [template]
// CHECK:STDOUT:   %ReturnIncomplete: %ReturnIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %CallTakeIncomplete.type: type = fn_type @CallTakeIncomplete [template]
// CHECK:STDOUT:   %CallTakeIncomplete: %CallTakeIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %CallReturnIncomplete.type: type = fn_type @CallReturnIncomplete [template]
// CHECK:STDOUT:   %CallReturnIncomplete: %CallReturnIncomplete.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
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
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.2] {}
// CHECK:STDOUT:   %CallClassFunction.decl: %CallClassFunction.type = fn_decl @CallClassFunction [template = constants.%CallClassFunction] {}
// CHECK:STDOUT:   %Class.ref.loc40: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %global_var.var: ref <error> = var global_var
// CHECK:STDOUT:   %global_var: ref <error> = bind_name global_var, %global_var.var
// CHECK:STDOUT:   %ConvertFromStruct.decl: %ConvertFromStruct.type = fn_decl @ConvertFromStruct [template = constants.%ConvertFromStruct] {
// CHECK:STDOUT:     %Class.ref.loc49: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @ConvertFromStruct.%return: ref %Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Class.ref.loc51: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc51_14: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc51_6.1: %.4 = param p
// CHECK:STDOUT:     @G.%p: %.4 = bind_name p, %p.loc51_6.1
// CHECK:STDOUT:     %int.make_type_32.loc51: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc51_20.1: type = value_of_initializer %int.make_type_32.loc51 [template = i32]
// CHECK:STDOUT:     %.loc51_20.2: type = converted %int.make_type_32.loc51, %.loc51_20.1 [template = i32]
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MemberAccess.decl: %MemberAccess.type = fn_decl @MemberAccess [template = constants.%MemberAccess] {
// CHECK:STDOUT:     %Class.ref.loc62: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc62_25: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc62_17.1: %.4 = param p
// CHECK:STDOUT:     @MemberAccess.%p: %.4 = bind_name p, %p.loc62_17.1
// CHECK:STDOUT:     %int.make_type_32.loc62: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc62_31.1: type = value_of_initializer %int.make_type_32.loc62 [template = i32]
// CHECK:STDOUT:     %.loc62_31.2: type = converted %int.make_type_32.loc62, %.loc62_31.1 [template = i32]
// CHECK:STDOUT:     @MemberAccess.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.decl: %Copy.type = fn_decl @Copy [template = constants.%Copy] {
// CHECK:STDOUT:     %Class.ref.loc80_12: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc80: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc80_9.1: %.4 = param p
// CHECK:STDOUT:     @Copy.%p: %.4 = bind_name p, %p.loc80_9.1
// CHECK:STDOUT:     %Class.ref.loc80_23: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @Copy.%return: ref %Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let.decl: %Let.type = fn_decl @Let [template = constants.%Let] {
// CHECK:STDOUT:     %Class.ref.loc84: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc84: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc84_8.1: %.4 = param p
// CHECK:STDOUT:     @Let.%p: %.4 = bind_name p, %p.loc84_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeIncomplete.decl: %TakeIncomplete.type = fn_decl @TakeIncomplete [template = constants.%TakeIncomplete] {
// CHECK:STDOUT:     %Class.ref.loc95: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc95_19.1: %Class = param c
// CHECK:STDOUT:     @TakeIncomplete.%c: %Class = bind_name c, %c.loc95_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnIncomplete.decl: %ReturnIncomplete.type = fn_decl @ReturnIncomplete [template = constants.%ReturnIncomplete] {
// CHECK:STDOUT:     %Class.ref.loc97: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @ReturnIncomplete.%return: ref %Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTakeIncomplete.decl: %CallTakeIncomplete.type = fn_decl @CallTakeIncomplete [template = constants.%CallTakeIncomplete] {
// CHECK:STDOUT:     %Class.ref.loc99: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc99: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:     %p.loc99_23.1: %.4 = param p
// CHECK:STDOUT:     @CallTakeIncomplete.%p: %.4 = bind_name p, %p.loc99_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallReturnIncomplete.decl: %CallReturnIncomplete.type = fn_decl @CallReturnIncomplete [template = constants.%CallReturnIncomplete] {}
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
// CHECK:STDOUT: fn @ConvertFromStruct() -> %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc49: %.3 = struct_literal ()
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: %.4) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc59: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberAccess(%p: %.4) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc70: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Copy(%p: %.4) -> %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc81: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let(%p: %.4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc92: ref %Class = deref %p.ref
// CHECK:STDOUT:   %c: <error> = bind_name c, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeIncomplete(%c: %Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnIncomplete() -> %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTakeIncomplete(%p: %.4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeIncomplete.ref.loc110: %TakeIncomplete.type = name_ref TakeIncomplete, file.%TakeIncomplete.decl [template = constants.%TakeIncomplete]
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc110: ref %Class = deref %p.ref
// CHECK:STDOUT:   %TakeIncomplete.call.loc110: init %.1 = call %TakeIncomplete.ref.loc110(<invalid>) [template = <error>]
// CHECK:STDOUT:   %TakeIncomplete.ref.loc122: %TakeIncomplete.type = name_ref TakeIncomplete, file.%TakeIncomplete.decl [template = constants.%TakeIncomplete]
// CHECK:STDOUT:   %.loc122: %.3 = struct_literal ()
// CHECK:STDOUT:   %TakeIncomplete.call.loc122: init %.1 = call %TakeIncomplete.ref.loc122(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallReturnIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnIncomplete.ref: %ReturnIncomplete.type = name_ref ReturnIncomplete, file.%ReturnIncomplete.decl [template = constants.%ReturnIncomplete]
// CHECK:STDOUT:   %ReturnIncomplete.call: init <error> = call %ReturnIncomplete.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
