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
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-12]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c.F();

  c.G();

  // This call is OK.
  (*p).F();

  (*p).G();
}

// CHECK:STDOUT: --- fail_addr_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %F.type.f1b: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.1f2: %F.type.f1b = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %c.patt: %Class = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:     %p.patt: %ptr.e71 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.e71 = value_param_pattern %p.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref.loc20_9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param1
// CHECK:STDOUT:     %.loc20: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc20_19: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc20_19 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type.f1b = fn_decl @F.1 [concrete = constants.%F.1f2] {
// CHECK:STDOUT:     %self.patt: %ptr.e71 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.e71 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc12_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc12_24: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = bind_name self, %self.param
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
// CHECK:STDOUT: fn @F.1[addr %self.param_patt: %ptr.e71]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.param_patt: %Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c.param_patt: %Class, %p.param_patt: %ptr.e71) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc28: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc28: %F.type.f1b = name_ref F, @Class.%F.decl [concrete = constants.%F.1f2]
// CHECK:STDOUT:   %F.bound.loc28: <bound method> = bound_method %c.ref.loc28, %F.ref.loc28
// CHECK:STDOUT:   %.loc28: ref %Class = temporary_storage
// CHECK:STDOUT:   %addr.loc28: %ptr.e71 = addr_of %.loc28
// CHECK:STDOUT:   %F.call.loc28: init %empty_tuple.type = call %F.bound.loc28(%addr.loc28)
// CHECK:STDOUT:   %c.ref.loc30: %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc30: %G.type = name_ref G, @Class.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.bound.loc30: <bound method> = bound_method %c.ref.loc30, %G.ref.loc30
// CHECK:STDOUT:   %G.call.loc30: init %empty_tuple.type = call %G.bound.loc30(%c.ref.loc30)
// CHECK:STDOUT:   %p.ref.loc33: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc33: ref %Class = deref %p.ref.loc33
// CHECK:STDOUT:   %F.ref.loc33: %F.type.f1b = name_ref F, @Class.%F.decl [concrete = constants.%F.1f2]
// CHECK:STDOUT:   %F.bound.loc33: <bound method> = bound_method %.loc33, %F.ref.loc33
// CHECK:STDOUT:   %addr.loc33: %ptr.e71 = addr_of %.loc33
// CHECK:STDOUT:   %F.call.loc33: init %empty_tuple.type = call %F.bound.loc33(%addr.loc33)
// CHECK:STDOUT:   %p.ref.loc35: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc35_4.1: ref %Class = deref %p.ref.loc35
// CHECK:STDOUT:   %G.ref.loc35: %G.type = name_ref G, @Class.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.bound.loc35: <bound method> = bound_method %.loc35_4.1, %G.ref.loc35
// CHECK:STDOUT:   %.loc35_4.2: %Class = bind_value %.loc35_4.1
// CHECK:STDOUT:   %G.call.loc35: init %empty_tuple.type = call %G.bound.loc35(%.loc35_4.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
