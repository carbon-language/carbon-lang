// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self/fail_ref_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self/fail_ref_self.carbon

class Class {
  fn F(ref self);
}

fn Make() -> Class;

fn F(c: Class, p: Class*) {
  // CHECK:STDERR: fail_ref_self.carbon:[[@LINE+7]]:3: error: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_ref_self.carbon:[[@LINE-9]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F(ref self);
  // CHECK:STDERR:        ^~~~~~~~
  // CHECK:STDERR:
  c.F();

  // This call is OK.
  (*p).F();

  // So is this one.
  Make().F();
}

// CHECK:STDOUT: --- fail_ref_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.f15: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %self.param_patt.b2b: %pattern_type.f15 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.123: %pattern_type.f15 = at_binding_pattern self, %self.param_patt.b2b [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.d76: Core.Form = init_form %Class [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.f15 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.f15 = return_slot_pattern %return.param_patt, %Class [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.f15 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.f15 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %ptr.7ee: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.8c5: type = pattern_type %ptr.7ee [concrete]
// CHECK:STDOUT:   %p.param_patt: %pattern_type.8c5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.8c5 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc35_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f15 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.f15 = return_slot_pattern %return.param_patt, %Class.ref [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %.loc19: Core.Form = init_form %Class.ref [concrete = constants.%.d76]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.f15 = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.f15 = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.8c5 = value_param_pattern [concrete = constants.%p.param_patt]
// CHECK:STDOUT:     %p.patt: %pattern_type.8c5 = at_binding_pattern p, %p.param_patt [concrete = constants.%p.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref.loc21_9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = wrapper_binding c, %c.param
// CHECK:STDOUT:     %p.param: %ptr.7ee = value_param call_param1
// CHECK:STDOUT:     %.loc21: type = splice_block %ptr [concrete = constants.%ptr.7ee] {
// CHECK:STDOUT:       %Class.ref.loc21_19: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc21_19 [concrete = constants.%ptr.7ee]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.7ee = wrapper_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f15 = ref_param_pattern [concrete = constants.%self.param_patt.b2b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.123]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: ref %Class = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F(%self.param: ref %Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> out %return.param: %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %Class, %p.param: %ptr.7ee) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc29: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound.loc29: <bound method> = bound_method %c.ref, %F.ref.loc29
// CHECK:STDOUT:   %Class.F.call.loc29: init %empty_tuple.type = call %Class.F.bound.loc29(<error>)
// CHECK:STDOUT:   %p.ref: %ptr.7ee = name_ref p, %p
// CHECK:STDOUT:   %.loc32: ref %Class = deref %p.ref
// CHECK:STDOUT:   %F.ref.loc32: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound.loc32: <bound method> = bound_method %.loc32, %F.ref.loc32
// CHECK:STDOUT:   %Class.F.call.loc32: init %empty_tuple.type = call %Class.F.bound.loc32(%.loc32)
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc35_8.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class to %.loc35_8.1 = call %Make.ref()
// CHECK:STDOUT:   %.loc35_8.2: ref %Class = temporary %.loc35_8.1, %Make.call
// CHECK:STDOUT:   %F.ref.loc35: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound.loc35: <bound method> = bound_method %.loc35_8.2, %F.ref.loc35
// CHECK:STDOUT:   %Class.F.call.loc35: init %empty_tuple.type = call %Class.F.bound.loc35(%.loc35_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc35_8.2, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc35_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_8.2(%self.param: ref %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
