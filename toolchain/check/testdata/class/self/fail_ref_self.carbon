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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.f15: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %self.param_patt.b2b: %pattern_type.f15 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f19: %pattern_type.f15 = wrapper_binding_pattern self, %self.param_patt.b2b [concrete]
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
// CHECK:STDOUT:   %c.patt: %pattern_type.f15 = wrapper_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %ptr.7ee: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.8c5: type = pattern_type %ptr.7ee [concrete]
// CHECK:STDOUT:   %p.param_patt: %pattern_type.8c5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.8c5 = wrapper_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc35_8.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3a1: %SubobjectDestroy.type = facet_value %Class, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.66c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3a1) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Class, (%DestroyImpl.impl_witness.66c, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.f15 = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.8c5 = value_param_pattern [concrete = constants.%p.param_patt]
// CHECK:STDOUT:     %p.patt: %pattern_type.8c5 = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt]
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
// CHECK:STDOUT:     %self.patt: %pattern_type.f15 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f19]
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
// CHECK:STDOUT:   %facet_value.loc35_8.1: %facet_type = facet_value constants.%Class, (constants.%DestroyImpl.impl_witness.66c, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc35_8.3: %facet_type = converted constants.%Class, %facet_value.loc35_8.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc35_8.2: %facet_type = facet_value constants.%Class, (constants.%DestroyImpl.impl_witness.66c, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc35_8.4: %facet_type = converted constants.%Class, %facet_value.loc35_8.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc35_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc35_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc35_8.2(%self.param: ref %Class) = "no_op";
// CHECK:STDOUT:
