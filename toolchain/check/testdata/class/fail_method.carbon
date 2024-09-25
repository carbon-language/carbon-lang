// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_method.carbon

class Class {
  fn NoSelf();
  fn WithSelf[self: Class]();
}

alias A = Class.WithSelf;

fn F(c: Class) {
  c.NoSelf();
  c.WithSelf();

  Class.NoSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+7]]:3: error: missing object argument in method call
  // CHECK:STDERR:   Class.WithSelf();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-13]]:3: note: calling function declared here
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Class.WithSelf();
  // CHECK:STDERR: fail_method.carbon:[[@LINE+7]]:3: error: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Class.WithSelf(c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-21]]:3: note: calling function declared here
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Class.WithSelf(c);

  // CHECK:STDERR: fail_method.carbon:[[@LINE+6]]:3: error: missing object argument in method call
  // CHECK:STDERR:   A();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_method.carbon:[[@LINE-30]]:3: note: calling function declared here
  // CHECK:STDERR:   fn WithSelf[self: Class]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  A();
}

// CHECK:STDOUT: --- fail_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %NoSelf.type: type = fn_type @NoSelf [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %NoSelf: %NoSelf.type = struct_value () [template]
// CHECK:STDOUT:   %WithSelf.type: type = fn_type @WithSelf [template]
// CHECK:STDOUT:   %WithSelf: %WithSelf.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Class.ref.loc16: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref: %WithSelf.type = name_ref WithSelf, @Class.%WithSelf.decl [template = constants.%WithSelf]
// CHECK:STDOUT:   %A: %WithSelf.type = bind_alias A, @Class.%WithSelf.decl [template = constants.%WithSelf]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Class.ref.loc18: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc18_6.1: %Class = param c, runtime_param0
// CHECK:STDOUT:     @F.%c: %Class = bind_name c, %c.loc18_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %NoSelf.decl: %NoSelf.type = fn_decl @NoSelf [template = constants.%NoSelf] {}
// CHECK:STDOUT:   %WithSelf.decl: %WithSelf.type = fn_decl @WithSelf [template = constants.%WithSelf] {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.loc13_15.1: %Class = param self, runtime_param0
// CHECK:STDOUT:     %self.loc13_15.2: %Class = bind_name self, %self.loc13_15.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .NoSelf = %NoSelf.decl
// CHECK:STDOUT:   .WithSelf = %WithSelf.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoSelf();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithSelf[@Class.%self.loc13_15.2: %Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c: %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc19: %Class = name_ref c, %c
// CHECK:STDOUT:   %NoSelf.ref.loc19: %NoSelf.type = name_ref NoSelf, @Class.%NoSelf.decl [template = constants.%NoSelf]
// CHECK:STDOUT:   %NoSelf.call.loc19: init %.1 = call %NoSelf.ref.loc19()
// CHECK:STDOUT:   %c.ref.loc20: %Class = name_ref c, %c
// CHECK:STDOUT:   %WithSelf.ref.loc20: %WithSelf.type = name_ref WithSelf, @Class.%WithSelf.decl [template = constants.%WithSelf]
// CHECK:STDOUT:   %.loc20: <bound method> = bound_method %c.ref.loc20, %WithSelf.ref.loc20
// CHECK:STDOUT:   %WithSelf.call.loc20: init %.1 = call %.loc20(%c.ref.loc20)
// CHECK:STDOUT:   %Class.ref.loc22: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %NoSelf.ref.loc22: %NoSelf.type = name_ref NoSelf, @Class.%NoSelf.decl [template = constants.%NoSelf]
// CHECK:STDOUT:   %NoSelf.call.loc22: init %.1 = call %NoSelf.ref.loc22()
// CHECK:STDOUT:   %Class.ref.loc30: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc30: %WithSelf.type = name_ref WithSelf, @Class.%WithSelf.decl [template = constants.%WithSelf]
// CHECK:STDOUT:   %WithSelf.call.loc30: init %.1 = call %WithSelf.ref.loc30(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Class.ref.loc38: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %WithSelf.ref.loc38: %WithSelf.type = name_ref WithSelf, @Class.%WithSelf.decl [template = constants.%WithSelf]
// CHECK:STDOUT:   %c.ref.loc38: %Class = name_ref c, %c
// CHECK:STDOUT:   %A.ref: %WithSelf.type = name_ref A, file.%A [template = constants.%WithSelf]
// CHECK:STDOUT:   %WithSelf.call.loc46: init %.1 = call %A.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
