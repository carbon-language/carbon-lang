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
  // CHECK:STDERR: fail_method.carbon:[[@LINE+7]]:3: ERROR: Missing object argument in method call.
  // CHECK:STDERR:   Class.WithSelf();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-13]]:3: Calling function declared here.
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Class.WithSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+7]]:3: ERROR: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Class.WithSelf(c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-21]]:3: Calling function declared here.
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Class.WithSelf(c);

  // CHECK:STDERR: fail_method.carbon:[[@LINE+6]]:3: ERROR: Missing object argument in method call.
  // CHECK:STDERR:   A();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-30]]:3: Calling function declared here.
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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Class.ref.loc12: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %A: <function> = bind_alias A, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Class.ref.loc14: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc14_6.1: Class = param c
// CHECK:STDOUT:     @F.%c: Class = bind_name c, %c.loc14_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %NoSelf: <function> = fn_decl @NoSelf [template] {}
// CHECK:STDOUT:   %WithSelf: <function> = fn_decl @WithSelf [template] {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.loc9_15.1: Class = param self
// CHECK:STDOUT:     %self.loc9_15.2: Class = bind_name self, %self.loc9_15.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
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
// CHECK:STDOUT:   %WithSelf.ref.loc16: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %.loc16_4: <bound method> = bound_method %c.ref.loc16, %WithSelf.ref.loc16
// CHECK:STDOUT:   %.loc16_13: init () = call %.loc16_4(%c.ref.loc16)
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %NoSelf.ref.loc18: <function> = name_ref NoSelf, @Class.%NoSelf [template = @Class.%NoSelf]
// CHECK:STDOUT:   %.loc18: init () = call %NoSelf.ref.loc18()
// CHECK:STDOUT:   %Class.ref.loc26: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc26: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %.loc26: init () = call %WithSelf.ref.loc26(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Class.ref.loc34: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc34: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %c.ref.loc34: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc34: init () = call %WithSelf.ref.loc34(<invalid>) [template = <error>]
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A [template = @Class.%WithSelf]
// CHECK:STDOUT:   %.loc42: init () = call %A.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
