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
  // CHECK:STDERR: fail_ref_self.carbon:[[@LINE+4]]:8: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:   fn F[ref self: Self]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[ref self: Self]();
}

fn Make() -> Class;

fn F(c: Class, p: Class*) {
  // CHECK:STDERR: fail_ref_self.carbon:[[@LINE+7]]:3: error: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_ref_self.carbon:[[@LINE-9]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[ref self: Self]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %pattern_type.138: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.23c: Core.Form = init_form %Class [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.bd7: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.9b1: type = pattern_type %ptr.bd7 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc39_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.138 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.138 = return_slot_pattern %return.param_patt, %Class.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %Class.ref [concrete = constants.%.23c]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.138 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.138 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.9b1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.9b1 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref.loc25_9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = value_binding c, %c.param
// CHECK:STDOUT:     %p.param: %ptr.bd7 = value_param call_param1
// CHECK:STDOUT:     %.loc25: type = splice_block %ptr [concrete = constants.%ptr.bd7] {
// CHECK:STDOUT:       %Class.ref.loc25_19: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc25_19 [concrete = constants.%ptr.bd7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.bd7 = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.138 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.138 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: ref %Class = ref_binding self, %self.param
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
// CHECK:STDOUT: fn @F(%c.param: %Class, %p.param: %ptr.bd7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc33: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound.loc33: <bound method> = bound_method %c.ref, %F.ref.loc33
// CHECK:STDOUT:   %Class.F.call.loc33: init %empty_tuple.type = call %Class.F.bound.loc33(<error>)
// CHECK:STDOUT:   %p.ref: %ptr.bd7 = name_ref p, %p
// CHECK:STDOUT:   %.loc36: ref %Class = deref %p.ref
// CHECK:STDOUT:   %F.ref.loc36: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound.loc36: <bound method> = bound_method %.loc36, %F.ref.loc36
// CHECK:STDOUT:   %Class.F.call.loc36: init %empty_tuple.type = call %Class.F.bound.loc36(%.loc36)
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc39_8.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class to %.loc39_8.1 = call %Make.ref()
// CHECK:STDOUT:   %.loc39_8.2: ref %Class = temporary %.loc39_8.1, %Make.call
// CHECK:STDOUT:   %F.ref.loc39: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound.loc39: <bound method> = bound_method %.loc39_8.2, %F.ref.loc39
// CHECK:STDOUT:   %Class.F.call.loc39: init %empty_tuple.type = call %Class.F.bound.loc39(%.loc39_8.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc39_8.2, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc39_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc39_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc39_8.2(%self.param: ref %Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
