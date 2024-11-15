// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_addr_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_addr_self.carbon

class Class {
  fn F[addr self: Class*]();
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+4]]:8: error: `addr` can only be applied to a binding with a pointer type [AddrOnNonPointerType]
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn G[addr self: Class]();
}

fn F(c: Class, p: Class*) {
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+6]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-12]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
  c.F();

  c.G();

  // This call is OK.
  (*p).F();

  (*p).G();
}

// CHECK:STDOUT: --- fail_addr_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Class [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %c.patt: %Class = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %p.patt: %.1 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.1 = value_param_pattern %p.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc20_9: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %Class.ref.loc20_19: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc20: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %c.param: %Class = value_param runtime_param0
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %p.param: %.1 = value_param runtime_param1
// CHECK:STDOUT:     %p: %.1 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc12_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc12_24: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %self.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.param: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self: %Class = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self.param_patt: %.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.param_patt: %Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c.param_patt: %Class, %p.param_patt: %.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc27: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc27: %F.type.1 = name_ref F, @Class.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc27: <bound method> = bound_method %c.ref.loc27, %F.ref.loc27
// CHECK:STDOUT:   %F.call.loc27: init %empty_tuple.type = call %.loc27(<error>)
// CHECK:STDOUT:   %c.ref.loc29: %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc29: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc29: <bound method> = bound_method %c.ref.loc29, %G.ref.loc29
// CHECK:STDOUT:   %G.call.loc29: init %empty_tuple.type = call %.loc29(%c.ref.loc29)
// CHECK:STDOUT:   %p.ref.loc32: %.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc32_4.1: ref %Class = deref %p.ref.loc32
// CHECK:STDOUT:   %F.ref.loc32: %F.type.1 = name_ref F, @Class.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc32_7: <bound method> = bound_method %.loc32_4.1, %F.ref.loc32
// CHECK:STDOUT:   %.loc32_4.2: %.1 = addr_of %.loc32_4.1
// CHECK:STDOUT:   %F.call.loc32: init %empty_tuple.type = call %.loc32_7(%.loc32_4.2)
// CHECK:STDOUT:   %p.ref.loc34: %.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc34_4.1: ref %Class = deref %p.ref.loc34
// CHECK:STDOUT:   %G.ref.loc34: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc34_7: <bound method> = bound_method %.loc34_4.1, %G.ref.loc34
// CHECK:STDOUT:   %.loc34_4.2: %Class = bind_value %.loc34_4.1
// CHECK:STDOUT:   %G.call.loc34: init %empty_tuple.type = call %.loc34_7(%.loc34_4.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
