// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn NoSelf();
  fn WithSelf[self: Class]();
}

alias A = Class.WithSelf;

fn F(c: Class) {
  c.NoSelf();
  c.WithSelf();

  Class.NoSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+6]]:3: ERROR: Missing object argument in method call.
  // CHECK:STDERR:   Class.WithSelf();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-13]]:3: Calling function declared here.
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  Class.WithSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+6]]:3: ERROR: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Class.WithSelf(c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-20]]:3: Calling function declared here.
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  Class.WithSelf(c);

  // CHECK:STDERR: fail_method.carbon:[[@LINE+6]]:3: ERROR: Missing object argument in method call.
  // CHECK:STDERR:   A();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-28]]:3: Calling function declared here.
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  A();
}

// CHECK:STDOUT: --- fail_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class {} [template = constants.%Class]
// CHECK:STDOUT:   %Class.ref.loc12: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %A: <function> = bind_alias A, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %F: <function> = fn_decl @F {
// CHECK:STDOUT:     %Class.ref.loc14: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc14_6.1: Class = param c
// CHECK:STDOUT:     @F.%c: Class = bind_name c, %c.loc14_6.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %NoSelf: <function> = fn_decl @NoSelf {} [template]
// CHECK:STDOUT:   %WithSelf: <function> = fn_decl @WithSelf {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.loc9_15.1: Class = param self
// CHECK:STDOUT:     %self.loc9_15.2: Class = bind_name self, %self.loc9_15.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .NoSelf = %NoSelf
// CHECK:STDOUT:   .WithSelf = %WithSelf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoSelf();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithSelf[@Class.%self.loc9_15.2: Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c: Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc15: Class = name_ref c, %c
// CHECK:STDOUT:   %NoSelf.ref.loc15: <function> = name_ref NoSelf, @Class.%NoSelf [template = @Class.%NoSelf]
// CHECK:STDOUT:   %.loc15: init () = call %NoSelf.ref.loc15()
// CHECK:STDOUT:   %c.ref.loc16: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc16_4: <bound method> = bound_method %c.ref.loc16, @Class.%WithSelf
// CHECK:STDOUT:   %.loc16_13: init () = call %.loc16_4(%c.ref.loc16)
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %NoSelf.ref.loc18: <function> = name_ref NoSelf, @Class.%NoSelf [template = @Class.%NoSelf]
// CHECK:STDOUT:   %.loc18: init () = call %NoSelf.ref.loc18()
// CHECK:STDOUT:   %Class.ref.loc25: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc25: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %.loc25: init () = call %WithSelf.ref.loc25(<invalid>)
// CHECK:STDOUT:   %Class.ref.loc32: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc32: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %c.ref.loc32: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc32: init () = call %WithSelf.ref.loc32(<invalid>)
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A [template = @Class.%WithSelf]
// CHECK:STDOUT:   %.loc40: init () = call %A.ref(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
