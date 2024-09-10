// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_addr_not_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_addr_not_self.carbon

class Class {
  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+4]]:8: ERROR: `addr` can only be applied to a `self` parameter.
  // CHECK:STDERR:   fn F[addr a: Class*]();
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR:
  fn F[addr a: Class*]();

  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+3]]:8: ERROR: `addr` can only be applied to a `self` parameter.
  // CHECK:STDERR:   fn G(addr b: Class*);
  // CHECK:STDERR:        ^~~~
  fn G(addr b: Class*);
}

// CHECK:STDOUT: --- fail_addr_not_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Class [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Class.ref.loc16: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc16: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %a.loc16_13.1: %.1 = param a
// CHECK:STDOUT:     %a.loc16_13.2: %.1 = bind_name a, %a.loc16_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Class.ref.loc21: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc21: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %b.loc21_13.1: %.1 = param b
// CHECK:STDOUT:     %b.loc21_13.2: %.1 = bind_name b, %b.loc21_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@Class.%a.loc16_13.2: %.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(@Class.%b.loc21_13.2: %.1);
// CHECK:STDOUT:
