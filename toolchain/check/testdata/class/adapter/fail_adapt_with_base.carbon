// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/fail_adapt_with_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/fail_adapt_with_base.carbon
class Simple {};
base class AdaptWithVirtual {
  virtual fn F[self: Self]();
  // CHECK:STDERR: fail_adapt_with_base.carbon:[[@LINE+7]]:3: error: adapter with virtual function [AdaptWithVirtual]
  // CHECK:STDERR:   adapt Simple;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_adapt_with_base.carbon:[[@LINE-4]]:3: note: first virtual function declaration is here [AdaptWithVirtualHere]
  // CHECK:STDERR:   virtual fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  adapt Simple;
}

// CHECK:STDOUT: --- fail_adapt_with_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Simple: type = class_type @Simple [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptWithVirtual: type = class_type @AdaptWithVirtual [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .AdaptWithVirtual = %AdaptWithVirtual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Simple.decl: type = class_decl @Simple [concrete = constants.%Simple] {} {}
// CHECK:STDOUT:   %AdaptWithVirtual.decl: type = class_decl @AdaptWithVirtual [concrete = constants.%AdaptWithVirtual] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Simple {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Simple
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptWithVirtual {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %AdaptWithVirtual = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AdaptWithVirtual = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AdaptWithVirtual = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AdaptWithVirtual [concrete = constants.%AdaptWithVirtual]
// CHECK:STDOUT:     %self: %AdaptWithVirtual = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple]
// CHECK:STDOUT:   adapt_decl %Simple.ref [concrete]
// CHECK:STDOUT:   complete_type_witness = <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptWithVirtual
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Simple = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F[%self.param_patt: %AdaptWithVirtual]();
// CHECK:STDOUT:
