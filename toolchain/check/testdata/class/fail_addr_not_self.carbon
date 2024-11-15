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
  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+4]]:8: error: `addr` can only be applied to a `self` parameter [AddrOnNonSelfParam]
  // CHECK:STDERR:   fn F[addr a:! Class*]();
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR:
  fn F[addr a:! Class*]();

  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+3]]:8: error: `addr` can only be applied to a `self` parameter [AddrOnNonSelfParam]
  // CHECK:STDERR:   fn G(addr b: Class*);
  // CHECK:STDERR:        ^~~~
  fn G(addr b: Class*);
}

// CHECK:STDOUT: --- fail_addr_not_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Class [template]
// CHECK:STDOUT:   %a: %.1 = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %.1 = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt.loc16_13.2: %.1 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc16_13.1 (constants.%a.patt)]
// CHECK:STDOUT:     %a.param_patt: %.1 = value_param_pattern %a.patt.loc16_13.2, runtime_param<invalid> [symbolic = %a.patt.loc16_13.1 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc16: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %a.param: %.1 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %a.loc16_13.2: %.1 = bind_symbolic_name a, 0, %a.param [symbolic = %a.loc16_13.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %.1 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.1 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc21: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %b.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.1 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%a.loc16_13.2: %.1) {
// CHECK:STDOUT:   %a.loc16_13.1: %.1 = bind_symbolic_name a, 0 [symbolic = %a.loc16_13.1 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc16_13.1: %.1 = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc16_13.1 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%a.param_patt: %.1]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %.1);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%a) {
// CHECK:STDOUT:   %a.loc16_13.1 => constants.%a
// CHECK:STDOUT:   %a.patt.loc16_13.1 => constants.%a
// CHECK:STDOUT: }
// CHECK:STDOUT:
