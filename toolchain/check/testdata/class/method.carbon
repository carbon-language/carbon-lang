// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %struct_type.k.1: type = struct_type {.k: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k.1 [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %CallAlias.type: type = fn_type @CallAlias [template]
// CHECK:STDOUT:   %CallAlias: %CallAlias.type = struct_value () [template]
// CHECK:STDOUT:   %CallOnConstBoundMethod.type: type = fn_type @CallOnConstBoundMethod [template]
// CHECK:STDOUT:   %CallOnConstBoundMethod: %CallOnConstBoundMethod.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.k.2: type = struct_type {.k: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%int_1.2) [template]
// CHECK:STDOUT:   %CallWithAddr.type: type = fn_type @CallWithAddr [template]
// CHECK:STDOUT:   %CallWithAddr: %CallWithAddr.type = struct_value () [template]
// CHECK:STDOUT:   %CallFThroughPointer.type: type = fn_type @CallFThroughPointer [template]
// CHECK:STDOUT:   %CallFThroughPointer: %CallFThroughPointer.type = struct_value () [template]
// CHECK:STDOUT:   %CallGThroughPointer.type: type = fn_type @CallGThroughPointer [template]
// CHECK:STDOUT:   %CallGThroughPointer: %CallGThroughPointer.type = struct_value () [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %CallFOnInitializingExpr.type: type = fn_type @CallFOnInitializingExpr [template]
// CHECK:STDOUT:   %CallFOnInitializingExpr: %CallFOnInitializingExpr.type = struct_value () [template]
// CHECK:STDOUT:   %CallGOnInitializingExpr.type: type = fn_type @CallGOnInitializingExpr [template]
// CHECK:STDOUT:   %CallGOnInitializingExpr: %CallGOnInitializingExpr.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
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
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Class = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %int_32.loc20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc20: init type = call constants.%Int(%int_32.loc20) [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_29.1: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_29.2: type = converted %int.make_type_signed.loc20, %.loc20_29.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc20: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc20: %Class = bind_name self, %self.param.loc20
// CHECK:STDOUT:     %return.param.loc20: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return.loc20: ref %i32 = return_slot %return.param.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %c.patt: %Class = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_22.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_22.2: type = converted %int.make_type_signed, %.loc24_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %c.param: %Class = value_param runtime_param0
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallAlias.decl: %CallAlias.type = fn_decl @CallAlias [template = constants.%CallAlias] {
// CHECK:STDOUT:     %c.patt: %Class = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_27.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_27.2: type = converted %int.make_type_signed, %.loc30_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %c.param: %Class = value_param runtime_param0
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallOnConstBoundMethod.decl: %CallOnConstBoundMethod.type = fn_decl @CallOnConstBoundMethod [template = constants.%CallOnConstBoundMethod] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc34_32.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc34_32.2: type = converted %int.make_type_signed, %.loc34_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWithAddr.decl: %CallWithAddr.type = fn_decl @CallWithAddr [template = constants.%CallWithAddr] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc38_22.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc38_22.2: type = converted %int.make_type_signed, %.loc38_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFThroughPointer.decl: %CallFThroughPointer.type = fn_decl @CallFThroughPointer [template = constants.%CallFThroughPointer] {
// CHECK:STDOUT:     %p.patt: %ptr.1 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.1 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %ptr: type = ptr_type %Class [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc43_38.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc43_38.2: type = converted %int.make_type_signed, %.loc43_38.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.1 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGThroughPointer.decl: %CallGThroughPointer.type = fn_decl @CallGThroughPointer [template = constants.%CallGThroughPointer] {
// CHECK:STDOUT:     %p.patt: %ptr.1 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.1 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %ptr: type = ptr_type %Class [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc47_38.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc47_38.2: type = converted %int.make_type_signed, %.loc47_38.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.1 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFOnInitializingExpr.decl: %CallFOnInitializingExpr.type = fn_decl @CallFOnInitializingExpr [template = constants.%CallFOnInitializingExpr] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc53_33.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc53_33.2: type = converted %int.make_type_signed, %.loc53_33.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGOnInitializingExpr.decl: %CallGOnInitializingExpr.type = fn_decl @CallGOnInitializingExpr [template = constants.%CallGOnInitializingExpr] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_33.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_33.2: type = converted %int.make_type_signed, %.loc57_33.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
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
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_25.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_25.2: type = converted %int.make_type_signed.loc12, %.loc12_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc12: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc12: %Class = bind_name self, %self.param.loc12
// CHECK:STDOUT:     %return.param.loc12: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return.loc12: ref %i32 = return_slot %return.param.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %ptr.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc13_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %ptr: type = ptr_type %Class [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_signed, %.loc13_31.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %self: %ptr.1 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, %F.decl [template = constants.%F]
// CHECK:STDOUT:   %A: %F.type = bind_alias A, %F.decl [template = constants.%F]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_10.2: type = converted %int.make_type_signed, %.loc17_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8: %Class.elem = field_decl k, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k.1 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .A = %A
// CHECK:STDOUT:   .k = %.loc17_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Class]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc20
// CHECK:STDOUT:   %k.ref: %Class.elem = name_ref k, @Class.%.loc17_8 [template = @Class.%.loc17_8]
// CHECK:STDOUT:   %.loc21_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc21_14.2: %i32 = bind_value %.loc21_14.1
// CHECK:STDOUT:   return %.loc21_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self.param_patt: %ptr.1]() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c.param_patt: %Class) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%c.ref)
// CHECK:STDOUT:   %.loc27_15.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc27_15.2: %i32 = converted %F.call, %.loc27_15.1
// CHECK:STDOUT:   return %.loc27_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAlias(%c.param_patt: %Class) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %A.ref: %F.type = name_ref A, @Class.%A [template = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %c.ref, %A.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%c.ref)
// CHECK:STDOUT:   %.loc31_15.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc31_15.2: %i32 = converted %F.call, %.loc31_15.1
// CHECK:STDOUT:   return %.loc31_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallOnConstBoundMethod() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc35_18.1: %struct_type.k.2 = struct_literal (%int_1)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc35_18.2: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc35_18.3: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc35_18.4: ref %i32 = class_element_access %.loc35_18.3, element0
// CHECK:STDOUT:   %.loc35_18.5: init %i32 = initialize_from %.loc35_18.2 to %.loc35_18.4 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc35_18.6: init %Class = class_init (%.loc35_18.5), %.loc35_18.3 [template = constants.%Class.val]
// CHECK:STDOUT:   %.loc35_18.7: ref %Class = temporary %.loc35_18.3, %.loc35_18.6
// CHECK:STDOUT:   %.loc35_20.1: ref %Class = converted %.loc35_18.1, %.loc35_18.7
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc35_20.1, %F.ref
// CHECK:STDOUT:   %.loc35_20.2: %Class = bind_value %.loc35_20.1
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%.loc35_20.2)
// CHECK:STDOUT:   %.loc35_33.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc35_33.2: %i32 = converted %F.call, %.loc35_33.1
// CHECK:STDOUT:   return %.loc35_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddr() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref %Class = var c
// CHECK:STDOUT:   %c: ref %Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %c.ref, %G.ref
// CHECK:STDOUT:   %addr: %ptr.1 = addr_of %c.ref
// CHECK:STDOUT:   %G.call: init %i32 = call %G.bound(%addr)
// CHECK:STDOUT:   %.loc40_15.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc40_15.2: %i32 = converted %G.call, %.loc40_15.1
// CHECK:STDOUT:   return %.loc40_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p.param_patt: %ptr.1) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc44_11.1: ref %Class = deref %p.ref
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc44_11.1, %F.ref
// CHECK:STDOUT:   %.loc44_11.2: %Class = bind_value %.loc44_11.1
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%.loc44_11.2)
// CHECK:STDOUT:   %.loc44_18.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc44_18.2: %i32 = converted %F.call, %.loc44_18.1
// CHECK:STDOUT:   return %.loc44_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p.param_patt: %ptr.1) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc48_11: ref %Class = deref %p.ref
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %.loc48_11, %G.ref
// CHECK:STDOUT:   %addr: %ptr.1 = addr_of %.loc48_11
// CHECK:STDOUT:   %G.call: init %i32 = call %G.bound(%addr)
// CHECK:STDOUT:   %.loc48_18.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc48_18.2: %i32 = converted %G.call, %.loc48_18.1
// CHECK:STDOUT:   return %.loc48_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc54_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc54_15.1
// CHECK:STDOUT:   %.loc54_15.2: ref %Class = temporary %.loc54_15.1, %Make.call
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc54_15.2, %F.ref
// CHECK:STDOUT:   %.loc54_15.3: %Class = bind_value %.loc54_15.2
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%.loc54_15.3)
// CHECK:STDOUT:   %.loc54_20.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc54_20.2: %i32 = converted %F.call, %.loc54_20.1
// CHECK:STDOUT:   return %.loc54_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc58_15.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc58_15.1
// CHECK:STDOUT:   %.loc58_15.2: ref %Class = temporary %.loc58_15.1, %Make.call
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %.loc58_15.2, %G.ref
// CHECK:STDOUT:   %addr: %ptr.1 = addr_of %.loc58_15.2
// CHECK:STDOUT:   %G.call: init %i32 = call %G.bound(%addr)
// CHECK:STDOUT:   %.loc58_20.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc58_20.2: %i32 = converted %G.call, %.loc58_20.1
// CHECK:STDOUT:   return %.loc58_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
