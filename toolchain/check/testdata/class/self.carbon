// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self.carbon

// --- self.carbon

library "[[@TEST_NAME]]";

class Class {
  fn F[self: Self]() -> i32;
  fn G[addr self: Self*]() -> i32;

  var n: i32;
}

fn Class.F[self: Self]() -> i32 {
  return self.n;
}

fn Class.G[addr self: Self*]() -> i32 {
  return (*self).n;
}

// --- fail_return_self_value.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_return_self_value.carbon:[[@LINE+6]]:25: error: cannot implicitly convert from `Class` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F[self: Self]() -> self;
  // CHECK:STDERR:                         ^~~~
  // CHECK:STDERR: fail_return_self_value.carbon:[[@LINE+3]]:25: note: type `Class` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F[self: Self]() -> self;
  // CHECK:STDERR:                         ^~~~
  fn F[self: Self]() -> self;
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %Class [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc11: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_29.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_29.2: type = converted %int.make_type_signed.loc11, %.loc11_29.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc11: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc11: %Class = bind_name self, %self.param.loc11
// CHECK:STDOUT:     %return.param.loc11: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return.loc11: ref %i32 = return_slot %return.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %ptr.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc15_12: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc15: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %ptr.loc15: type = ptr_type %Class [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_35.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_35.2: type = converted %int.make_type_signed.loc15, %.loc15_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc15: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc15: %ptr.1 = bind_name self, %self.param.loc15
// CHECK:STDOUT:     %return.param.loc15: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return.loc15: ref %i32 = return_slot %return.param.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_25.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_25.2: type = converted %int.make_type_signed.loc5, %.loc5_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc5: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc5: %Class = bind_name self, %self.param.loc5
// CHECK:STDOUT:     %return.param.loc5: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return.loc5: ref %i32 = return_slot %return.param.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %ptr.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc15_12: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc6: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %ptr.loc6: type = ptr_type %Class [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_31.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_31.2: type = converted %int.make_type_signed.loc6, %.loc6_31.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc6: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc6: %ptr.1 = bind_name self, %self.param.loc6
// CHECK:STDOUT:     %return.param.loc6: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return.loc6: ref %i32 = return_slot %return.param.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.2: type = converted %int.make_type_signed, %.loc8_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8: %Class.elem = field_decl n, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc8_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Class]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc11
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc8_8 [template = @Class.%.loc8_8]
// CHECK:STDOUT:   %.loc12_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc12_14.2: %i32 = bind_value %.loc12_14.1
// CHECK:STDOUT:   return %.loc12_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self.param_patt: %ptr.1]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.1 = name_ref self, %self.loc15
// CHECK:STDOUT:   %.loc16_11: ref %Class = deref %self.ref
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc8_8 [template = @Class.%.loc8_8]
// CHECK:STDOUT:   %.loc16_17.1: ref %i32 = class_element_access %.loc16_11, element0
// CHECK:STDOUT:   %.loc16_17.2: %i32 = bind_value %.loc16_17.1
// CHECK:STDOUT:   return %.loc16_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_self_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
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
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.ref: %Class = name_ref self, %self
// CHECK:STDOUT:     %.loc11: type = converted %self.ref, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self: %Class = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Class]() -> <error>;
// CHECK:STDOUT:
