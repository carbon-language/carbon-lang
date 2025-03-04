// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self_type.carbon

class Class {
  fn F[self: Self]() -> i32;
  fn Make() -> Self {
    returned var s: Self;
    s = {.p = &s};
    return var;
  }
  var p: Self*;
}

fn Class.F[self: Self]() -> i32 {
  return (*self.p).F();
}

// CHECK:STDOUT: --- self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %ptr.e71 [concrete]
// CHECK:STDOUT:   %struct_type.p: type = struct_type {.p: %ptr.e71} [concrete]
// CHECK:STDOUT:   %complete_type.56d: <witness> = complete_type_witness %struct_type.p [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param.loc21: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc21: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc21: %Class = bind_name self, %self.param.loc21
// CHECK:STDOUT:     %return.param.loc21: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc21: ref %i32 = return_slot %return.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param.loc12: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc12: %Class = bind_name self, %self.param.loc12
// CHECK:STDOUT:     %return.param.loc12: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc12: ref %i32 = return_slot %return.param.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_8: %Class.elem = field_decl p, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc18_3: %Class.elem = var_pattern %.loc18_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Class.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.p [concrete = constants.%complete_type.56d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .p = %.loc18_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Class]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc21
// CHECK:STDOUT:   %p.ref: %Class.elem = name_ref p, @Class.%.loc18_8 [concrete = @Class.%.loc18_8]
// CHECK:STDOUT:   %.loc22_16.1: ref %ptr.e71 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc22_16.2: %ptr.e71 = bind_value %.loc22_16.1
// CHECK:STDOUT:   %.loc22_11.1: ref %Class = deref %.loc22_16.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc22_11.1, %F.ref
// CHECK:STDOUT:   %.loc22_11.2: %Class = bind_value %.loc22_11.1
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%.loc22_11.2)
// CHECK:STDOUT:   %.loc22_23.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc22_23.2: %i32 = converted %F.call, %.loc22_23.1
// CHECK:STDOUT:   return %.loc22_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %Class = binding_pattern s
// CHECK:STDOUT:     %.loc14: %Class = var_pattern %s.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.ref.loc14: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:   %s: ref %Class = bind_name s, %return
// CHECK:STDOUT:   %s.ref.loc15_5: ref %Class = name_ref s, %s
// CHECK:STDOUT:   %s.ref.loc15_16: ref %Class = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.e71 = addr_of %s.ref.loc15_16
// CHECK:STDOUT:   %.loc15_17.1: %struct_type.p = struct_literal (%addr)
// CHECK:STDOUT:   %.loc15_17.2: ref %ptr.e71 = class_element_access %s.ref.loc15_5, element0
// CHECK:STDOUT:   %.loc15_17.3: init %ptr.e71 = initialize_from %addr to %.loc15_17.2
// CHECK:STDOUT:   %.loc15_17.4: init %Class = class_init (%.loc15_17.3), %s.ref.loc15_5
// CHECK:STDOUT:   %.loc15_7: init %Class = converted %.loc15_17.1, %.loc15_17.4
// CHECK:STDOUT:   assign %s.ref.loc15_5, %.loc15_7
// CHECK:STDOUT:   return %s to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
