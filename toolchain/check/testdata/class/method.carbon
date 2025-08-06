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
  fn G[addr self: Self*]() -> i32;

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

fn CallWithAddr() -> i32 {
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
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G [concrete]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b40: <witness> = impl_witness @Class.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.k.0bf: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.954: <witness> = complete_type_witness %struct_type.k.0bf [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallAlias.type: type = fn_type @CallAlias [concrete]
// CHECK:STDOUT:   %CallAlias: %CallAlias.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallOnConstBoundMethod.type: type = fn_type @CallOnConstBoundMethod [concrete]
// CHECK:STDOUT:   %CallOnConstBoundMethod: %CallOnConstBoundMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.k.240: type = struct_type {.k: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %CallWithAddr.type: type = fn_type @CallWithAddr [concrete]
// CHECK:STDOUT:   %CallWithAddr: %CallWithAddr.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFThroughPointer.type: type = fn_type @CallFThroughPointer [concrete]
// CHECK:STDOUT:   %CallFThroughPointer: %CallFThroughPointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallGThroughPointer.type: type = fn_type @CallGThroughPointer [concrete]
// CHECK:STDOUT:   %CallGThroughPointer: %CallGThroughPointer.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallAlias = %CallAlias.decl
// CHECK:STDOUT:     .CallOnConstBoundMethod = %CallOnConstBoundMethod.decl
// CHECK:STDOUT:     .CallWithAddr = %CallWithAddr.decl
// CHECK:STDOUT:     .CallFThroughPointer = %CallFThroughPointer.decl
// CHECK:STDOUT:     .CallGThroughPointer = %CallGThroughPointer.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .CallFOnInitializingExpr = %CallFOnInitializingExpr.decl
// CHECK:STDOUT:     .CallGOnInitializingExpr = %CallGOnInitializingExpr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
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
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.761 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallAlias.decl: %CallAlias.type = fn_decl @CallAlias [concrete = constants.%CallAlias] {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.761 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallOnConstBoundMethod.decl: %CallOnConstBoundMethod.type = fn_decl @CallOnConstBoundMethod [concrete = constants.%CallOnConstBoundMethod] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWithAddr.decl: %CallWithAddr.type = fn_decl @CallWithAddr [concrete = constants.%CallWithAddr] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFThroughPointer.decl: %CallFThroughPointer.type = fn_decl @CallFThroughPointer [concrete = constants.%CallFThroughPointer] {
// CHECK:STDOUT:     %p.patt: %pattern_type.796 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.796 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc47: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGThroughPointer.decl: %CallGThroughPointer.type = fn_decl @CallGThroughPointer [concrete = constants.%CallGThroughPointer] {
// CHECK:STDOUT:     %p.patt: %pattern_type.796 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.796 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc51: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.761 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.761 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFOnInitializingExpr.decl: %CallFOnInitializingExpr.type = fn_decl @CallFOnInitializingExpr [concrete = constants.%CallFOnInitializingExpr] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
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
// CHECK:STDOUT: impl @Class.as.Destroy.impl: constants.%Class as constants.%Destroy.type {
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.decl: %Class.as.Destroy.impl.Op.type = fn_decl @Class.as.Destroy.impl.Op [concrete = constants.%Class.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.796 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param.loc16: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc16: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc16: %Class = bind_name self, %self.param.loc16
// CHECK:STDOUT:     %return.param.loc16: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc16: ref %i32 = return_slot %return.param.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.796 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc17_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc17_23: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, %Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %A: %Class.F.type = bind_alias A, %Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc21: %Class.elem = field_decl k, element0 [concrete]
// CHECK:STDOUT:   impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.b40]
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
// CHECK:STDOUT: fn @Class.F(%self.param.loc24: %Class) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc24
// CHECK:STDOUT:   %k.ref: %Class.elem = name_ref k, @Class.%.loc21 [concrete = @Class.%.loc21]
// CHECK:STDOUT:   %.loc25_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc25_14.2: %i32 = bind_value %.loc25_14.1
// CHECK:STDOUT:   return %.loc25_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G(%self.param: %ptr.e71) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.as.Destroy.impl.Op(%self.param: %ptr.e71) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c.param: %Class) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%c.ref)
// CHECK:STDOUT:   %.loc31_15.1: %i32 = value_of_initializer %Class.F.call
// CHECK:STDOUT:   %.loc31_15.2: %i32 = converted %Class.F.call, %.loc31_15.1
// CHECK:STDOUT:   return %.loc31_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAlias(%c.param: %Class) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %A.ref: %Class.F.type = name_ref A, @Class.%A [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %c.ref, %A.ref
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%c.ref)
// CHECK:STDOUT:   %.loc35_15.1: %i32 = value_of_initializer %Class.F.call
// CHECK:STDOUT:   %.loc35_15.2: %i32 = converted %Class.F.call, %.loc35_15.1
// CHECK:STDOUT:   return %.loc35_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallOnConstBoundMethod() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc39_18.1: %struct_type.k.240 = struct_literal (%int_1)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc39_18.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_18.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc39_18.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc39_18.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc39_18.3: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc39_18.4: ref %i32 = class_element_access %.loc39_18.3, element0
// CHECK:STDOUT:   %.loc39_18.5: init %i32 = initialize_from %.loc39_18.2 to %.loc39_18.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc39_18.6: init %Class = class_init (%.loc39_18.5), %.loc39_18.3 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc39_18.7: ref %Class = temporary %.loc39_18.3, %.loc39_18.6
// CHECK:STDOUT:   %.loc39_20.1: ref %Class = converted %.loc39_18.1, %.loc39_18.7
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %.loc39_20.1, %F.ref
// CHECK:STDOUT:   %.loc39_20.2: %Class = bind_value %.loc39_20.1
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%.loc39_20.2)
// CHECK:STDOUT:   %.loc39_33.1: %i32 = value_of_initializer %Class.F.call
// CHECK:STDOUT:   %.loc39_33.2: %i32 = converted %Class.F.call, %.loc39_33.1
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc39_18.3, constants.%Class.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.e71 = addr_of %.loc39_18.3
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Class.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %.loc39_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddr() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.761 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Class = var %c.var_patt
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %c: ref %Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound: <bound method> = bound_method %c.ref, %G.ref
// CHECK:STDOUT:   %addr.loc44: %ptr.e71 = addr_of %c.ref
// CHECK:STDOUT:   %Class.G.call: init %i32 = call %Class.G.bound(%addr.loc44)
// CHECK:STDOUT:   %.loc44_15.1: %i32 = value_of_initializer %Class.G.call
// CHECK:STDOUT:   %.loc44_15.2: %i32 = converted %Class.G.call, %.loc44_15.1
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%Class.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc43: %ptr.e71 = addr_of %c.var
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Class.as.Destroy.impl.Op.bound(%addr.loc43)
// CHECK:STDOUT:   return %.loc44_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p.param: %ptr.e71) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc48_11.1: ref %Class = deref %p.ref
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %.loc48_11.1, %F.ref
// CHECK:STDOUT:   %.loc48_11.2: %Class = bind_value %.loc48_11.1
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%.loc48_11.2)
// CHECK:STDOUT:   %.loc48_18.1: %i32 = value_of_initializer %Class.F.call
// CHECK:STDOUT:   %.loc48_18.2: %i32 = converted %Class.F.call, %.loc48_18.1
// CHECK:STDOUT:   return %.loc48_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p.param: %ptr.e71) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc52_11: ref %Class = deref %p.ref
// CHECK:STDOUT:   %G.ref: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound: <bound method> = bound_method %.loc52_11, %G.ref
// CHECK:STDOUT:   %addr: %ptr.e71 = addr_of %.loc52_11
// CHECK:STDOUT:   %Class.G.call: init %i32 = call %Class.G.bound(%addr)
// CHECK:STDOUT:   %.loc52_18.1: %i32 = value_of_initializer %Class.G.call
// CHECK:STDOUT:   %.loc52_18.2: %i32 = converted %Class.G.call, %.loc52_18.1
// CHECK:STDOUT:   return %.loc52_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc58_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc58_15.1
// CHECK:STDOUT:   %.loc58_15.2: ref %Class = temporary %.loc58_15.1, %Make.call
// CHECK:STDOUT:   %F.ref: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound: <bound method> = bound_method %.loc58_15.2, %F.ref
// CHECK:STDOUT:   %.loc58_15.3: %Class = bind_value %.loc58_15.2
// CHECK:STDOUT:   %Class.F.call: init %i32 = call %Class.F.bound(%.loc58_15.3)
// CHECK:STDOUT:   %.loc58_20.1: %i32 = value_of_initializer %Class.F.call
// CHECK:STDOUT:   %.loc58_20.2: %i32 = converted %Class.F.call, %.loc58_20.1
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc58_15.1, constants.%Class.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.e71 = addr_of %.loc58_15.1
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Class.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %.loc58_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc62_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc62_15.1
// CHECK:STDOUT:   %.loc62_15.2: ref %Class = temporary %.loc62_15.1, %Make.call
// CHECK:STDOUT:   %G.ref: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound: <bound method> = bound_method %.loc62_15.2, %G.ref
// CHECK:STDOUT:   %addr.loc62_15.1: %ptr.e71 = addr_of %.loc62_15.2
// CHECK:STDOUT:   %Class.G.call: init %i32 = call %Class.G.bound(%addr.loc62_15.1)
// CHECK:STDOUT:   %.loc62_20.1: %i32 = value_of_initializer %Class.G.call
// CHECK:STDOUT:   %.loc62_20.2: %i32 = converted %Class.G.call, %.loc62_20.1
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc62_15.1, constants.%Class.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc62_15.2: %ptr.e71 = addr_of %.loc62_15.1
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Class.as.Destroy.impl.Op.bound(%addr.loc62_15.2)
// CHECK:STDOUT:   return %.loc62_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
