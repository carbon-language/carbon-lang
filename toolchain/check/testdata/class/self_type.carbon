// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param.loc24: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc24: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc24: %Class = bind_name self, %self.param.loc24
// CHECK:STDOUT:     %return.param.loc24: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc24: ref %i32 = return_slot %return.param.loc24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param.loc15: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc15: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc15: %Class = bind_name self, %self.param.loc15
// CHECK:STDOUT:     %return.param.loc15: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc15: ref %i32 = return_slot %return.param.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.761 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.761 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:   %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:   %.loc21: %Class.elem = field_decl p, element0 [concrete]
// CHECK:STDOUT:   %struct_type.p: type = struct_type {.p: %ptr.e71} [concrete = constants.%struct_type.p]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.p [concrete = constants.%complete_type.56d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .p = %.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param.loc24: %Class) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc24
// CHECK:STDOUT:   %p.ref: %Class.elem = name_ref p, @Class.%.loc21 [concrete = @Class.%.loc21]
// CHECK:STDOUT:   %.loc25_16.1: ref %ptr.e71 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc25_16.2: %ptr.e71 = bind_value %.loc25_16.1
// CHECK:STDOUT:   %.loc25_11.1: ref %Class = deref %.loc25_16.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc25_11.1, %F.ref
// CHECK:STDOUT:   %.loc25_11.2: %Class = bind_value %.loc25_11.1
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%.loc25_11.2)
// CHECK:STDOUT:   %.loc25_23.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc25_23.2: %i32 = converted %F.call, %.loc25_23.1
// CHECK:STDOUT:   return %.loc25_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.761 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.761 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.ref.loc17: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:   %s: ref %Class = bind_name s, %return
// CHECK:STDOUT:   %s.ref.loc18_5: ref %Class = name_ref s, %s
// CHECK:STDOUT:   %s.ref.loc18_16: ref %Class = name_ref s, %s
// CHECK:STDOUT:   %addr: %ptr.e71 = addr_of %s.ref.loc18_16
// CHECK:STDOUT:   %.loc18_17.1: %struct_type.p = struct_literal (%addr)
// CHECK:STDOUT:   %.loc18_17.2: ref %ptr.e71 = class_element_access %s.ref.loc18_5, element0
// CHECK:STDOUT:   %.loc18_17.3: init %ptr.e71 = initialize_from %addr to %.loc18_17.2
// CHECK:STDOUT:   %.loc18_17.4: init %Class = class_init (%.loc18_17.3), %s.ref.loc18_5
// CHECK:STDOUT:   %.loc18_7: init %Class = converted %.loc18_17.1, %.loc18_17.4
// CHECK:STDOUT:   assign %s.ref.loc18_5, %.loc18_7
// CHECK:STDOUT:   return %s to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
