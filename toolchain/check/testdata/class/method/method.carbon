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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method.carbon

class Class {
  fn F[self: Self]() -> i32;
  fn G[ref self: Self]() -> i32;

  alias A = F;

  var k: i32;
}

fn Class.F[self: Self]() -> i32 {
  return self.k;
}

fn Call(c: Class) -> i32 {
  // TODO: The sem-ir for this call doesn't distinguish the `self` argument from
  // the explicit arguments.
  return c.F();
}

fn CallAlias(c: Class) -> i32 {
  return c.A();
}

fn CallOnConstBoundMethod() -> i32 {
  return ({.k = 1} as Class).F();
}

fn CallWithRef() -> i32 {
  var c: Class;
  return c.G();
}

fn CallFThroughPointer(p: Class*) -> i32 {
  return (*p).F();
}

fn CallGThroughPointer(p: Class*) -> i32 {
  return (*p).G();
}

fn Make() -> Class;

fn CallFOnInitializingExpr() -> i32 {
  return Make().F();
}

fn CallGOnInitializingExpr() -> i32 {
  return Make().G();
}

// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.904: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.e54: Core.Form = init_form %i32, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G [concrete]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k.0bf: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.954: <witness> = complete_type_witness %struct_type.k.0bf [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallAlias.type: type = fn_type @CallAlias [concrete]
// CHECK:STDOUT:   %CallAlias: %CallAlias.type = struct_value () [concrete]
// CHECK:STDOUT:   %.437: Core.Form = init_form %i32, call_param0 [concrete]
// CHECK:STDOUT:   %CallOnConstBoundMethod.type: type = fn_type @CallOnConstBoundMethod [concrete]
// CHECK:STDOUT:   %CallOnConstBoundMethod: %CallOnConstBoundMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.k.240: type = struct_type {.k: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.k.240 = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallWithRef.type: type = fn_type @CallWithRef [concrete]
// CHECK:STDOUT:   %CallWithRef: %CallWithRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.8e5: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.018: type = pattern_type %ptr.8e5 [concrete]
// CHECK:STDOUT:   %CallFThroughPointer.type: type = fn_type @CallFThroughPointer [concrete]
// CHECK:STDOUT:   %CallFThroughPointer: %CallFThroughPointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGThroughPointer.type: type = fn_type @CallGThroughPointer [concrete]
// CHECK:STDOUT:   %CallGThroughPointer: %CallGThroughPointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ddd: Core.Form = init_form %Class, call_param0 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFOnInitializingExpr.type: type = fn_type @CallFOnInitializingExpr [concrete]
// CHECK:STDOUT:   %CallFOnInitializingExpr: %CallFOnInitializingExpr.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGOnInitializingExpr.type: type = fn_type @CallGOnInitializingExpr [concrete]
// CHECK:STDOUT:   %CallGOnInitializingExpr: %CallGOnInitializingExpr.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallAlias = %CallAlias.decl
// CHECK:STDOUT:     .CallOnConstBoundMethod = %CallOnConstBoundMethod.decl
// CHECK:STDOUT:     .CallWithRef = %CallWithRef.decl
// CHECK:STDOUT:     .CallFThroughPointer = %CallFThroughPointer.decl
// CHECK:STDOUT:     .CallGThroughPointer = %CallGThroughPointer.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .CallFOnInitializingExpr = %CallFOnInitializingExpr.decl
// CHECK:STDOUT:     .CallGOnInitializingExpr = %CallGOnInitializingExpr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.904 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.904 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc24: Core.Form = init_form %i32.loc24, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %self.param.loc24: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc24: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc24: %Class = value_binding self, %self.param.loc24
// CHECK:STDOUT:     %return.param.loc24: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc24: ref %i32 = return_slot %return.param.loc24
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %c.patt: %pattern_type.904 = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.904 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc28: Core.Form = init_form %i32, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallAlias.decl: %CallAlias.type = fn_decl @CallAlias [concrete = constants.%CallAlias] {
// CHECK:STDOUT:     %c.patt: %pattern_type.904 = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.904 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc34: Core.Form = init_form %i32, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallOnConstBoundMethod.decl: %CallOnConstBoundMethod.type = fn_decl @CallOnConstBoundMethod [concrete = constants.%CallOnConstBoundMethod] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc38: Core.Form = init_form %i32, call_param0 [concrete = constants.%.437]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWithRef.decl: %CallWithRef.type = fn_decl @CallWithRef [concrete = constants.%CallWithRef] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc42: Core.Form = init_form %i32, call_param0 [concrete = constants.%.437]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFThroughPointer.decl: %CallFThroughPointer.type = fn_decl @CallFThroughPointer [concrete = constants.%CallFThroughPointer] {
// CHECK:STDOUT:     %p.patt: %pattern_type.018 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.018 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc47_38: Core.Form = init_form %i32, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %p.param: %ptr.8e5 = value_param call_param0
// CHECK:STDOUT:     %.loc47_32: type = splice_block %ptr [concrete = constants.%ptr.8e5] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.8e5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.8e5 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGThroughPointer.decl: %CallGThroughPointer.type = fn_decl @CallGThroughPointer [concrete = constants.%CallGThroughPointer] {
// CHECK:STDOUT:     %p.patt: %pattern_type.018 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.018 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc51_38: Core.Form = init_form %i32, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %p.param: %ptr.8e5 = value_param call_param0
// CHECK:STDOUT:     %.loc51_32: type = splice_block %ptr [concrete = constants.%ptr.8e5] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.8e5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.8e5 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.904 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.904 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %.loc55: Core.Form = init_form %Class.ref, call_param0 [concrete = constants.%.ddd]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFOnInitializingExpr.decl: %CallFOnInitializingExpr.type = fn_decl @CallFOnInitializingExpr [concrete = constants.%CallFOnInitializingExpr] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc57: Core.Form = init_form %i32, call_param0 [concrete = constants.%.437]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGOnInitializingExpr.decl: %CallGOnInitializingExpr.type = fn_decl @CallGOnInitializingExpr [concrete = constants.%CallGOnInitializingExpr] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc61: Core.Form = init_form %i32, call_param0 [concrete = constants.%.437]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.904 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.904 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %i32.loc16, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %self.param.loc16: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc16: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc16: %Class = value_binding self, %self.param.loc16
// CHECK:STDOUT:     %return.param.loc16: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc16: ref %i32 = return_slot %return.param.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.904 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.904 = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc17: Core.Form = init_form %i32, call_param1 [concrete = constants.%.e54]
// CHECK:STDOUT:     %self.param: ref %Class = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: ref %Class = ref_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, %Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %A: %Class.F.type = alias_binding A, %Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc21: %Class.elem = field_decl k, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.k.0bf [concrete = constants.%complete_type.954]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT:   .G = %Class.G.decl
// CHECK:STDOUT:   .A = %A
// CHECK:STDOUT:   .k = %.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F(%self.param.loc24: %Class) -> out %return.param.loc24: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc24
// CHECK:STDOUT:   %k.ref: %Class.elem = name_ref k, @Class.%.loc21 [concrete = @Class.%.loc21]
// CHECK:STDOUT:   %.loc25_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc25_14.2: %i32 = acquire_value %.loc25_14.1
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc25_14.1: <bound method> = bound_method %.loc25_14.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_14.2: <bound method> = bound_method %.loc25_14.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc25_14.2(%.loc25_14.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G(%self.param: ref %Class) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c.param: %Class) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%c.ref)
// CHECK:STDOUT:   return %Class.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAlias(%c.param: %Class) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %A.ref: %Class.F.type = name_ref A, @Class.%A [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %c.ref, %A.ref
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%c.ref)
// CHECK:STDOUT:   return %Class.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallOnConstBoundMethod() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc39_18.1: %struct_type.k.240 = struct_literal (%int_1) [concrete = constants.%struct]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc39_18.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_18.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc39_18.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc39_18.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc39_18.3: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc39_18.4: ref %i32 = class_element_access %.loc39_18.3, element0
// CHECK:STDOUT:   %.loc39_18.5: init %i32 to %.loc39_18.4 = in_place_init %.loc39_18.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc39_18.6: init %Class to %.loc39_18.3 = class_init (%.loc39_18.5) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc39_18.7: ref %Class = temporary %.loc39_18.3, %.loc39_18.6
// CHECK:STDOUT:   %.loc39_20.1: ref %Class = converted %.loc39_18.1, %.loc39_18.7
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %.loc39_20.1, %F.ref
// CHECK:STDOUT:   %.loc39_20.2: %Class = acquire_value %.loc39_20.1
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%.loc39_20.2)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc39_18.7, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc39_18.7)
// CHECK:STDOUT:   return %Class.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %Class) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithRef() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.904 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.904 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Class = var %c.var_patt
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %c: ref %Class = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound: <bound method> = bound_method %c.ref, %G.ref
// CHECK:STDOUT:   %Class.G.call: init %i32 = call %Class.G.bound(%c.ref)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %c.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%c.var)
// CHECK:STDOUT:   return %Class.G.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p.param: %ptr.8e5) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.8e5 = name_ref p, %p
// CHECK:STDOUT:   %.loc48_11.1: ref %Class = deref %p.ref
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %.loc48_11.1, %F.ref
// CHECK:STDOUT:   %.loc48_11.2: %Class = acquire_value %.loc48_11.1
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%.loc48_11.2)
// CHECK:STDOUT:   return %Class.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p.param: %ptr.8e5) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.8e5 = name_ref p, %p
// CHECK:STDOUT:   %.loc52: ref %Class = deref %p.ref
// CHECK:STDOUT:   %G.ref: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound: <bound method> = bound_method %.loc52, %G.ref
// CHECK:STDOUT:   %Class.G.call: init %i32 = call %Class.G.bound(%.loc52)
// CHECK:STDOUT:   return %Class.G.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> out %return.param: %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc58_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class to %.loc58_15.1 = call %Make.ref()
// CHECK:STDOUT:   %.loc58_15.2: ref %Class = temporary %.loc58_15.1, %Make.call
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %.loc58_15.2, %F.ref
// CHECK:STDOUT:   %.loc58_15.3: %Class = acquire_value %.loc58_15.2
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%.loc58_15.3)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc58_15.2, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc58_15.2)
// CHECK:STDOUT:   return %Class.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc62_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class to %.loc62_15.1 = call %Make.ref()
// CHECK:STDOUT:   %.loc62_15.2: ref %Class = temporary %.loc62_15.1, %Make.call
// CHECK:STDOUT:   %G.ref: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound: <bound method> = bound_method %.loc62_15.2, %G.ref
// CHECK:STDOUT:   %Class.G.call: init %i32 = call %Class.G.bound(%.loc62_15.2)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc62_15.2, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc62_15.2)
// CHECK:STDOUT:   return %Class.G.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
