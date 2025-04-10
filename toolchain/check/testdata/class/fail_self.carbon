// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_self.carbon

class Class {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:8: error: `self` can only be declared in an implicit parameter list [SelfOutsideImplicitParamList]
  // CHECK:STDERR:   fn F(self: Self);
  // CHECK:STDERR:        ^~~~~~~~~~
  // CHECK:STDERR:
  fn F(self: Self);

  fn G() -> Self;
}

// CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:12: error: `self` can only be declared in an implicit parameter list [SelfOutsideImplicitParamList]
// CHECK:STDERR: fn Class.F(self: Self) {
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
fn Class.F(self: Self) {
}

fn Class.G() -> Self {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:7: error: `self` can only be declared in an implicit parameter list [SelfOutsideImplicitParamList]
  // CHECK:STDERR:   var self: Self;
  // CHECK:STDERR:       ^~~~~~~~~~
  // CHECK:STDERR:
  var self: Self;
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:10: error: cannot copy value of type `Class` [CopyOfUncopyableType]
  // CHECK:STDERR:   return self;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  return self;
}

class WrongSelf {
  fn F[self: Class]();
}

fn CallWrongSelf(ws: WrongSelf) {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `WrongSelf` to `Class` [ConversionFailure]
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_self.carbon:[[@LINE+7]]:3: note: type `WrongSelf` does not implement interface `Core.ImplicitAs(Class)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_self.carbon:[[@LINE-10]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[self: Class]();
  // CHECK:STDERR:        ^~~~~~~~~~~
  // CHECK:STDERR:
  ws.F();
}

// CHECK:STDOUT: --- fail_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %F.type.f1b: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.1f2: %F.type.f1b = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %WrongSelf: type = class_type @WrongSelf [concrete]
// CHECK:STDOUT:   %F.type.25f: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.3a3: %F.type.25f = struct_value () [concrete]
// CHECK:STDOUT:   %CallWrongSelf.type: type = fn_type @CallWrongSelf [concrete]
// CHECK:STDOUT:   %CallWrongSelf: %CallWrongSelf.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .WrongSelf = %WrongSelf.decl
// CHECK:STDOUT:     .CallWrongSelf = %CallWrongSelf.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.f1b = fn_decl @F.1 [concrete = constants.%F.1f2] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc25: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc25: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc25: %Class = bind_name self, %self.param.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc28: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %return.param.loc28: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return.loc28: ref %Class = return_slot %return.param.loc28
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongSelf.decl: type = class_decl @WrongSelf [concrete = constants.%WrongSelf] {} {}
// CHECK:STDOUT:   %CallWrongSelf.decl: %CallWrongSelf.type = fn_decl @CallWrongSelf [concrete = constants.%CallWrongSelf] {
// CHECK:STDOUT:     %ws.patt: %WrongSelf = binding_pattern ws
// CHECK:STDOUT:     %ws.param_patt: %WrongSelf = value_param_pattern %ws.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ws.param: %WrongSelf = value_param call_param0
// CHECK:STDOUT:     %WrongSelf.ref: type = name_ref WrongSelf, file.%WrongSelf.decl [concrete = constants.%WrongSelf]
// CHECK:STDOUT:     %ws: %WrongSelf = bind_name ws, %ws.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type.f1b = fn_decl @F.1 [concrete = constants.%F.1f2] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc16: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc16: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc16: %Class = bind_name self, %self.param.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %return.param.loc18: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return.loc18: ref %Class = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongSelf {
// CHECK:STDOUT:   %F.decl: %F.type.25f = fn_decl @F.2 [concrete = constants.%F.3a3] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%WrongSelf
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1(%self.param_patt: %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param_patt: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %.loc33: %Class = var_pattern %self.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %self.var: ref %Class = var self
// CHECK:STDOUT:   %Self.ref.loc33: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:   %self: ref %Class = bind_name self, %self.var
// CHECK:STDOUT:   %self.ref: ref %Class = name_ref self, %self
// CHECK:STDOUT:   %.loc38: %Class = bind_value %self.ref
// CHECK:STDOUT:   return <error> to %return.loc28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongSelf(%ws.param_patt: %WrongSelf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ws.ref: %WrongSelf = name_ref ws, %ws
// CHECK:STDOUT:   %F.ref: %F.type.25f = name_ref F, @WrongSelf.%F.decl [concrete = constants.%F.3a3]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %ws.ref, %F.ref
// CHECK:STDOUT:   %.loc56: %Class = converted %ws.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
