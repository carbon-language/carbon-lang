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
  // CHECK:STDERR: fail_self.carbon:[[@LINE+9]]:3: error: cannot implicitly convert from `WrongSelf` to `Class` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_self.carbon:[[@LINE+6]]:3: note: type `WrongSelf` does not implement interface `ImplicitAs(Class)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_self.carbon:[[@LINE-10]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[self: Class]();
  // CHECK:STDERR:        ^~~~~~~~~~~
  ws.F();
}

// CHECK:STDOUT: --- fail_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %WrongSelf: type = class_type @WrongSelf [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %CallWrongSelf.type: type = fn_type @CallWrongSelf [template]
// CHECK:STDOUT:   %CallWrongSelf: %CallWrongSelf.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .WrongSelf = %WrongSelf.decl
// CHECK:STDOUT:     .CallWrongSelf = %CallWrongSelf.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc25: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.param.loc25: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc25: %Class = bind_name self, %self.param.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc28: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %return.param.loc28: ref %Class = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param.loc28
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongSelf.decl: type = class_decl @WrongSelf [template = constants.%WrongSelf] {} {}
// CHECK:STDOUT:   %CallWrongSelf.decl: %CallWrongSelf.type = fn_decl @CallWrongSelf [template = constants.%CallWrongSelf] {
// CHECK:STDOUT:     %ws.patt: %WrongSelf = binding_pattern ws
// CHECK:STDOUT:     %ws.param_patt: %WrongSelf = value_param_pattern %ws.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %WrongSelf.ref: type = name_ref WrongSelf, file.%WrongSelf.decl [template = constants.%WrongSelf]
// CHECK:STDOUT:     %ws.param: %WrongSelf = value_param runtime_param0
// CHECK:STDOUT:     %ws: %WrongSelf = bind_name ws, %ws.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.param.loc16: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc16: %Class = bind_name self, %self.param.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %return.param.loc18: ref %Class = out_param runtime_param0
// CHECK:STDOUT:     %.loc18: ref %Class = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongSelf {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.param: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self: %Class = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc43: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%WrongSelf
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1(%self.param_patt: %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref.loc33: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %self.var: ref %Class = var self
// CHECK:STDOUT:   %self: ref %Class = bind_name self, %self.var
// CHECK:STDOUT:   %self.ref: ref %Class = name_ref self, %self
// CHECK:STDOUT:   %.loc38: %Class = bind_value %self.ref
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongSelf(%ws.param_patt: %WrongSelf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ws.ref: %WrongSelf = name_ref ws, %ws
// CHECK:STDOUT:   %F.ref: %F.type.2 = name_ref F, @WrongSelf.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %.loc55_5: <bound method> = bound_method %ws.ref, %F.ref
// CHECK:STDOUT:   %.loc55_3: %Class = converted %ws.ref, <error> [template = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc55_5(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
