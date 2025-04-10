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

  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+4]]:8: error: `addr` can only be applied to a `self` parameter [AddrOnNonSelfParam]
  // CHECK:STDERR:   fn G(addr b: Class*);
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR:
  fn G(addr b: Class*);
}

// CHECK:STDOUT: --- fail_addr_not_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %a: %ptr = bind_symbolic_name a, 0 [symbolic]
// CHECK:STDOUT:   %a.patt: %ptr = symbolic_binding_pattern a, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt.loc16_13.1: %ptr = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc16_13.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.loc16_13.2: %ptr = bind_symbolic_name a, 0 [symbolic = %a.loc16_13.1 (constants.%a)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.patt: %ptr = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %ptr = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc22: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %ptr = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%a.loc16_13.2: %ptr) {
// CHECK:STDOUT:   %a.loc16_13.1: %ptr = bind_symbolic_name a, 0 [symbolic = %a.loc16_13.1 (constants.%a)]
// CHECK:STDOUT:   %a.patt.loc16_13.2: %ptr = symbolic_binding_pattern a, 0 [symbolic = %a.patt.loc16_13.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%a.patt.loc16_13.1: %ptr]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%a) {
// CHECK:STDOUT:   %a.loc16_13.1 => constants.%a
// CHECK:STDOUT:   %a.patt.loc16_13.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
