// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_call_ref_method_with_initializing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_call_ref_method_with_initializing.carbon

class Class {
  fn G[ref self: Self]() -> i32;

  var k: i32;
}

fn Make() -> Class;

fn CallGOnInitializingExpr() -> i32 {
  // TODO: Either make this work or delete this TODO, depending on how #5545
  // resolves this issue.
  // CHECK:STDERR: fail_call_ref_method_with_initializing.carbon:[[@LINE+7]]:10: error: cannot bind durable reference to non-reference value of type `Class` [ConversionFailureNonRefToRef]
  // CHECK:STDERR:   return Make().G();
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_call_ref_method_with_initializing.carbon:[[@LINE-13]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn G[ref self: Self]() -> i32;
  // CHECK:STDERR:        ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Make().G();
}

// CHECK:STDOUT: --- fail_call_ref_method_with_initializing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G [concrete]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.954: <witness> = complete_type_witness %struct_type.k [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGOnInitializingExpr.type: type = fn_type @CallGOnInitializingExpr [concrete]
// CHECK:STDOUT:   %CallGOnInitializingExpr: %CallGOnInitializingExpr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Class, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.179: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.635: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.179 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.635, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .CallGOnInitializingExpr = %CallGOnInitializingExpr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.761 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.761 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGOnInitializingExpr.decl: %CallGOnInitializingExpr.type = fn_decl @CallGOnInitializingExpr [concrete = constants.%CallGOnInitializingExpr] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: ref %Class = ref_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc18: %Class.elem = field_decl k, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.k [concrete = constants.%complete_type.954]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .G = %Class.G.decl
// CHECK:STDOUT:   .k = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G(%self.param: %Class) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc33_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc33_15.1
// CHECK:STDOUT:   %.loc33_15.2: ref %Class = temporary %.loc33_15.1, %Make.call
// CHECK:STDOUT:   %G.ref: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound: <bound method> = bound_method %.loc33_15.2, %G.ref
// CHECK:STDOUT:   %.loc33_15.3: %Class = bind_value %.loc33_15.2
// CHECK:STDOUT:   %self.var: ref %Class = var @Class.G.%self.param_patt
// CHECK:STDOUT:   %Class.G.call: init %i32 = call %Class.G.bound(%self.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc33_15.1: <bound method> = bound_method %self.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.635
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.635, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_15.1: <bound method> = bound_method %self.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc33_15.1: %ptr.e71 = addr_of %self.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc33_15.1: init %empty_tuple.type = call %bound_method.loc33_15.1(%addr.loc33_15.1)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc33_15.2: <bound method> = bound_method %.loc33_15.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.635
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.635, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_15.2: <bound method> = bound_method %.loc33_15.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc33_15.2: %ptr.e71 = addr_of %.loc33_15.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc33_15.2: init %empty_tuple.type = call %bound_method.loc33_15.2(%addr.loc33_15.2)
// CHECK:STDOUT:   return %Class.G.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
