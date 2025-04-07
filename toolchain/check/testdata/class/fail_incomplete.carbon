// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_incomplete.carbon

// --- fail_forward_decl.carbon

library "[[@TEST_NAME]]";

class Class;

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:4: error: cannot declare a member of incomplete class `Class` [QualifiedDeclInIncompleteClassScope]
// CHECK:STDERR: fn Class.Function() {}
// CHECK:STDERR:    ^~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn Class.Function() {}

fn CallClassFunction() {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:3: error: member access into incomplete class `Class` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR:   Class.Function();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-15]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  Class.Function();
}

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:17: error: binding pattern has incomplete type `Class` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var global_var: Class;
// CHECK:STDERR:                 ^~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-25]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var global_var: Class;

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:24: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn ConvertFromStruct() -> Class { return {}; }
// CHECK:STDERR:                        ^~~~~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-34]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn ConvertFromStruct() -> Class { return {}; }

fn G(p: Class*) -> i32 {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:10: error: member access into object of incomplete type `Class` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   return p->n;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-44]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  return p->n;
}

fn MemberAccess(p: Class*) -> i32 {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:11: error: member access into object of incomplete type `Class` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   return (*p).n;
  // CHECK:STDERR:           ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-55]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  return (*p).n;
}

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:20: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn Copy(p: Class*) -> Class {
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-65]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn Copy(p: Class*) -> Class {
  return *p;
}

fn Let(p: Class*) {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:10: error: binding pattern has incomplete type `Class` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   let c: Class = *p;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-77]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  let c: Class = *p;
}

fn TakeIncomplete(c: Class);

fn ReturnIncomplete() -> Class;

fn CallTakeIncomplete(p: Class*) {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:18: error: forming value of incomplete type `Class` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   TakeIncomplete(*p);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-92]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-11]]:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete(*p);

  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:18: error: forming value of incomplete type `Class` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   TakeIncomplete({});
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-104]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-23]]:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete({});
}

fn CallReturnIncomplete() {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:3: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnIncomplete();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-118]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-35]]:23: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnIncomplete() -> Class;
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR:
  ReturnIncomplete();
}

class IncompleteAddrSelf {
  fn F[addr self: Class*]();
}

fn CallIncompleteAddrSelf(p: Class*) {
  // TODO: Should this be valid?
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:3: error: invalid use of incomplete type `Class` [IncompleteTypeInConversion]
  // CHECK:STDERR:   p->(IncompleteAddrSelf.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-137]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-11]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->(IncompleteAddrSelf.F)();
}

// --- fail_in_definition.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE+7]]:10: error: field has incomplete type `C` [IncompleteTypeInFieldDecl]
  // CHECK:STDERR:   var c: C;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE-4]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~~~~~
  // CHECK:STDERR:
  var c: C;
}

// CHECK:STDOUT: --- fail_forward_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallClassFunction.type: type = fn_type @CallClassFunction [concrete]
// CHECK:STDOUT:   %CallClassFunction: %CallClassFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertFromStruct.type: type = fn_type @ConvertFromStruct [concrete]
// CHECK:STDOUT:   %ConvertFromStruct: %ConvertFromStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %MemberAccess.type: type = fn_type @MemberAccess [concrete]
// CHECK:STDOUT:   %MemberAccess: %MemberAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [concrete]
// CHECK:STDOUT:   %Copy: %Copy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Let.type: type = fn_type @Let [concrete]
// CHECK:STDOUT:   %Let: %Let.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeIncomplete.type: type = fn_type @TakeIncomplete [concrete]
// CHECK:STDOUT:   %TakeIncomplete: %TakeIncomplete.type = struct_value () [concrete]
// CHECK:STDOUT:   %ReturnIncomplete.type: type = fn_type @ReturnIncomplete [concrete]
// CHECK:STDOUT:   %ReturnIncomplete: %ReturnIncomplete.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallTakeIncomplete.type: type = fn_type @CallTakeIncomplete [concrete]
// CHECK:STDOUT:   %CallTakeIncomplete: %CallTakeIncomplete.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallReturnIncomplete.type: type = fn_type @CallReturnIncomplete [concrete]
// CHECK:STDOUT:   %CallReturnIncomplete: %CallReturnIncomplete.type = struct_value () [concrete]
// CHECK:STDOUT:   %IncompleteAddrSelf: type = class_type @IncompleteAddrSelf [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %CallIncompleteAddrSelf.type: type = fn_type @CallIncompleteAddrSelf [concrete]
// CHECK:STDOUT:   %CallIncompleteAddrSelf: %CallIncompleteAddrSelf.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .CallClassFunction = %CallClassFunction.decl
// CHECK:STDOUT:     .global_var = %global_var
// CHECK:STDOUT:     .ConvertFromStruct = %ConvertFromStruct.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .MemberAccess = %MemberAccess.decl
// CHECK:STDOUT:     .Copy = %Copy.decl
// CHECK:STDOUT:     .Let = %Let.decl
// CHECK:STDOUT:     .TakeIncomplete = %TakeIncomplete.decl
// CHECK:STDOUT:     .ReturnIncomplete = %ReturnIncomplete.decl
// CHECK:STDOUT:     .CallTakeIncomplete = %CallTakeIncomplete.decl
// CHECK:STDOUT:     .CallReturnIncomplete = %CallReturnIncomplete.decl
// CHECK:STDOUT:     .IncompleteAddrSelf = %IncompleteAddrSelf.decl
// CHECK:STDOUT:     .CallIncompleteAddrSelf = %CallIncompleteAddrSelf.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {} {}
// CHECK:STDOUT:   %CallClassFunction.decl: %CallClassFunction.type = fn_decl @CallClassFunction [concrete = constants.%CallClassFunction] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %global_var.patt: <error> = binding_pattern global_var
// CHECK:STDOUT:     %.loc33: <error> = var_pattern %global_var.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %global_var.var: ref <error> = var global_var
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %global_var: <error> = bind_name global_var, <error>
// CHECK:STDOUT:   %ConvertFromStruct.decl: %ConvertFromStruct.type = fn_decl @ConvertFromStruct [concrete = constants.%ConvertFromStruct] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %ptr.e71 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.e71 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc44: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MemberAccess.decl: %MemberAccess.type = fn_decl @MemberAccess [concrete = constants.%MemberAccess] {
// CHECK:STDOUT:     %p.patt: %ptr.e71 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.e71 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc55: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.decl: %Copy.type = fn_decl @Copy [concrete = constants.%Copy] {
// CHECK:STDOUT:     %p.patt: %ptr.e71 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.e71 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc73_23: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc73: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc73_12: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc73_12 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param1
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let.decl: %Let.type = fn_decl @Let [concrete = constants.%Let] {
// CHECK:STDOUT:     %p.patt: %ptr.e71 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.e71 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc77: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc77: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc77 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeIncomplete.decl: %TakeIncomplete.type = fn_decl @TakeIncomplete [concrete = constants.%TakeIncomplete] {
// CHECK:STDOUT:     %c.patt: %Class = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnIncomplete.decl: %ReturnIncomplete.type = fn_decl @ReturnIncomplete [concrete = constants.%ReturnIncomplete] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTakeIncomplete.decl: %CallTakeIncomplete.type = fn_decl @CallTakeIncomplete [concrete = constants.%CallTakeIncomplete] {
// CHECK:STDOUT:     %p.patt: %ptr.e71 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.e71 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc92: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallReturnIncomplete.decl: %CallReturnIncomplete.type = fn_decl @CallReturnIncomplete [concrete = constants.%CallReturnIncomplete] {} {}
// CHECK:STDOUT:   %IncompleteAddrSelf.decl: type = class_decl @IncompleteAddrSelf [concrete = constants.%IncompleteAddrSelf] {} {}
// CHECK:STDOUT:   %CallIncompleteAddrSelf.decl: %CallIncompleteAddrSelf.type = fn_decl @CallIncompleteAddrSelf [concrete = constants.%CallIncompleteAddrSelf] {
// CHECK:STDOUT:     %p.patt: %ptr.e71 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.e71 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc136: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: class @IncompleteAddrSelf {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %ptr.e71 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.e71 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc133_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc133_24: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%IncompleteAddrSelf
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallClassFunction() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %Function.ref: <error> = name_ref Function, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertFromStruct() -> %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc42: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %ptr.e71) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc52: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberAccess(%p.param_patt: %ptr.e71) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc63: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Copy(%p.param_patt: %ptr.e71) -> %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc74: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let(%p.param_patt: %ptr.e71) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: <error> = binding_pattern c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc85: ref %Class = deref %p.ref
// CHECK:STDOUT:   %Class.ref.loc85: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %c: <error> = bind_name c, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeIncomplete(%c.param_patt: %Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnIncomplete() -> %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTakeIncomplete(%p.param_patt: %ptr.e71) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeIncomplete.ref.loc103: %TakeIncomplete.type = name_ref TakeIncomplete, file.%TakeIncomplete.decl [concrete = constants.%TakeIncomplete]
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc103: ref %Class = deref %p.ref
// CHECK:STDOUT:   %TakeIncomplete.call.loc103: init %empty_tuple.type = call %TakeIncomplete.ref.loc103(<error>)
// CHECK:STDOUT:   %TakeIncomplete.ref.loc115: %TakeIncomplete.type = name_ref TakeIncomplete, file.%TakeIncomplete.decl [concrete = constants.%TakeIncomplete]
// CHECK:STDOUT:   %.loc115: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %TakeIncomplete.call.loc115: init %empty_tuple.type = call %TakeIncomplete.ref.loc115(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallReturnIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnIncomplete.ref: %ReturnIncomplete.type = name_ref ReturnIncomplete, file.%ReturnIncomplete.decl [concrete = constants.%ReturnIncomplete]
// CHECK:STDOUT:   %ReturnIncomplete.call: init <error> = call %ReturnIncomplete.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.param_patt: %ptr.e71]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIncompleteAddrSelf(%p.param_patt: %ptr.e71) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %IncompleteAddrSelf.ref: type = name_ref IncompleteAddrSelf, file.%IncompleteAddrSelf.decl [concrete = constants.%IncompleteAddrSelf]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @IncompleteAddrSelf.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc148: ref %Class = deref %p.ref
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc148, %F.ref
// CHECK:STDOUT:   %addr: <error> = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc12_8: <error> = field_decl c, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc12_3: <error> = var_pattern %.loc12_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref <error> = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .c = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
