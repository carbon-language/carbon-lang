// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn NoSelf();
  fn WithSelf[self: Class]();
}

fn F(c: Class) {
  c.NoSelf();
  c.WithSelf();

  Class.NoSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+6]]:3: ERROR: Missing object argument in method call.
  // CHECK:STDERR:   Class.WithSelf();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-11]]:3: Calling function declared here.
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  Class.WithSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+6]]:3: ERROR: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Class.WithSelf(c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-18]]:3: Calling function declared here.
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  Class.WithSelf(c);
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
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl, .F = %F} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %NoSelf: <function> = fn_decl @NoSelf [template]
// CHECK:STDOUT:   %WithSelf: <function> = fn_decl @WithSelf [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .NoSelf = %NoSelf
// CHECK:STDOUT:   .WithSelf = %WithSelf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoSelf();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithSelf[%self: Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c: Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc13: Class = name_ref c, %c
// CHECK:STDOUT:   %NoSelf.ref.loc13: <function> = name_ref NoSelf, @Class.%NoSelf [template = @Class.%NoSelf]
// CHECK:STDOUT:   %.loc13: init () = call %NoSelf.ref.loc13()
// CHECK:STDOUT:   %c.ref.loc14: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc14_4: <bound method> = bound_method %c.ref.loc14, @Class.%WithSelf
// CHECK:STDOUT:   %.loc14_13: init () = call %.loc14_4(%c.ref.loc14)
// CHECK:STDOUT:   %Class.ref.loc16: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %NoSelf.ref.loc16: <function> = name_ref NoSelf, @Class.%NoSelf [template = @Class.%NoSelf]
// CHECK:STDOUT:   %.loc16: init () = call %NoSelf.ref.loc16()
// CHECK:STDOUT:   %Class.ref.loc23: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc23: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %.loc23: init () = call %WithSelf.ref.loc23(<invalid>)
// CHECK:STDOUT:   %Class.ref.loc30: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc30: <function> = name_ref WithSelf, @Class.%WithSelf [template = @Class.%WithSelf]
// CHECK:STDOUT:   %c.ref.loc30: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc30: init () = call %WithSelf.ref.loc30(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
