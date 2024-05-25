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
// CHECK:STDOUT:   %NoSelf: type = fn_type @NoSelf [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: NoSelf = struct_value () [template]
// CHECK:STDOUT:   %WithSelf: type = fn_type @WithSelf [template]
// CHECK:STDOUT:   %struct.2: WithSelf = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Class.ref.loc12: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref: WithSelf = name_ref WithSelf, @Class.%WithSelf.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %A: WithSelf = bind_alias A, @Class.%WithSelf.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.3] {
// CHECK:STDOUT:     %Class.ref.loc14: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc14_6.1: Class = param c
// CHECK:STDOUT:     @F.%c: Class = bind_name c, %c.loc14_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %NoSelf.decl: NoSelf = fn_decl @NoSelf [template = constants.%struct.1] {}
// CHECK:STDOUT:   %WithSelf.decl: WithSelf = fn_decl @WithSelf [template = constants.%struct.2] {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.loc9_15.1: Class = param self
// CHECK:STDOUT:     %self.loc9_15.2: Class = bind_name self, %self.loc9_15.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .NoSelf = %NoSelf.decl
// CHECK:STDOUT:   .WithSelf = %WithSelf.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoSelf();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithSelf[@Class.%self.loc9_15.2: Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c: Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc15: Class = name_ref c, %c
// CHECK:STDOUT:   %NoSelf.ref.loc15: NoSelf = name_ref NoSelf, @Class.%NoSelf.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %NoSelf.call.loc15: init () = call %NoSelf.ref.loc15()
// CHECK:STDOUT:   %c.ref.loc16: Class = name_ref c, %c
// CHECK:STDOUT:   %WithSelf.ref.loc16: WithSelf = name_ref WithSelf, @Class.%WithSelf.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc16: <bound method> = bound_method %c.ref.loc16, %WithSelf.ref.loc16
// CHECK:STDOUT:   %WithSelf.call.loc16: init () = call %.loc16(%c.ref.loc16)
// CHECK:STDOUT:   %Class.ref.loc18: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %NoSelf.ref.loc18: NoSelf = name_ref NoSelf, @Class.%NoSelf.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %NoSelf.call.loc18: init () = call %NoSelf.ref.loc18()
// CHECK:STDOUT:   %Class.ref.loc26: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc26: WithSelf = name_ref WithSelf, @Class.%WithSelf.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %WithSelf.call.loc26: init () = call %WithSelf.ref.loc26(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Class.ref.loc34: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc34: WithSelf = name_ref WithSelf, @Class.%WithSelf.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %c.ref.loc34: Class = name_ref c, %c
// CHECK:STDOUT:   %WithSelf.call.loc34: init () = call %WithSelf.ref.loc34(<invalid>) [template = <error>]
// CHECK:STDOUT:   %A.ref: WithSelf = name_ref A, file.%A [template = constants.%struct.2]
// CHECK:STDOUT:   %WithSelf.call.loc42: init () = call %A.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
