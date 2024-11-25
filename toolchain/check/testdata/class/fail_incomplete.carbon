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

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:17: error: variable has incomplete type `Class` [IncompleteTypeInVarDecl]
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
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:10: error: `let` binding has incomplete type `Class` [IncompleteTypeInLetDecl]
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
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
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE+6]]:10: error: field has incomplete type `C` [IncompleteTypeInVarDecl]
  // CHECK:STDERR:   var c: C;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE-4]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~~~~~
  var c: C;
}

// CHECK:STDOUT: --- fail_forward_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.1: %.type = struct_value () [template]
// CHECK:STDOUT:   %CallClassFunction.type: type = fn_type @CallClassFunction [template]
// CHECK:STDOUT:   %CallClassFunction: %CallClassFunction.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertFromStruct.type: type = fn_type @ConvertFromStruct [template]
// CHECK:STDOUT:   %ConvertFromStruct: %ConvertFromStruct.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %Class [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %MemberAccess.type: type = fn_type @MemberAccess [template]
// CHECK:STDOUT:   %MemberAccess: %MemberAccess.type = struct_value () [template]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [template]
// CHECK:STDOUT:   %Copy: %Copy.type = struct_value () [template]
// CHECK:STDOUT:   %Let.type: type = fn_type @Let [template]
// CHECK:STDOUT:   %Let: %Let.type = struct_value () [template]
// CHECK:STDOUT:   %TakeIncomplete.type: type = fn_type @TakeIncomplete [template]
// CHECK:STDOUT:   %TakeIncomplete: %TakeIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnIncomplete.type: type = fn_type @ReturnIncomplete [template]
// CHECK:STDOUT:   %ReturnIncomplete: %ReturnIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %CallTakeIncomplete.type: type = fn_type @CallTakeIncomplete [template]
// CHECK:STDOUT:   %CallTakeIncomplete: %CallTakeIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %CallReturnIncomplete.type: type = fn_type @CallReturnIncomplete [template]
// CHECK:STDOUT:   %CallReturnIncomplete: %CallReturnIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %IncompleteAddrSelf: type = class_type @IncompleteAddrSelf [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %CallIncompleteAddrSelf.type: type = fn_type @CallIncompleteAddrSelf [template]
// CHECK:STDOUT:   %CallIncompleteAddrSelf: %CallIncompleteAddrSelf.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
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
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.1] {} {}
// CHECK:STDOUT:   %CallClassFunction.decl: %CallClassFunction.type = fn_decl @CallClassFunction [template = constants.%CallClassFunction] {} {}
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %global_var.var: ref <error> = var global_var
// CHECK:STDOUT:   %global_var: ref <error> = bind_name global_var, %global_var.var
// CHECK:STDOUT:   %ConvertFromStruct.decl: %ConvertFromStruct.type = fn_decl @ConvertFromStruct [template = constants.%ConvertFromStruct] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc44_14: type = ptr_type %Class [template = constants.%.3]
// CHECK:STDOUT:     %.loc44_20.1: Core.IntLiteral = int_value 32 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc44_20.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc44_20.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc44_20.3: type = converted %int.make_type_signed, %.loc44_20.2 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.3 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MemberAccess.decl: %MemberAccess.type = fn_decl @MemberAccess [template = constants.%MemberAccess] {
// CHECK:STDOUT:     %p.patt: %.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc55_25: type = ptr_type %Class [template = constants.%.3]
// CHECK:STDOUT:     %.loc55_31.1: Core.IntLiteral = int_value 32 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc55_31.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc55_31.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc55_31.3: type = converted %int.make_type_signed, %.loc55_31.2 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.3 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.decl: %Copy.type = fn_decl @Copy [template = constants.%Copy] {
// CHECK:STDOUT:     %p.patt: %.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc73_12: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc73_17: type = ptr_type %Class [template = constants.%.3]
// CHECK:STDOUT:     %Class.ref.loc73_23: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %p.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.3 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let.decl: %Let.type = fn_decl @Let [template = constants.%Let] {
// CHECK:STDOUT:     %p.patt: %.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc77: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc77: type = ptr_type %Class [template = constants.%.3]
// CHECK:STDOUT:     %p.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.3 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeIncomplete.decl: %TakeIncomplete.type = fn_decl @TakeIncomplete [template = constants.%TakeIncomplete] {
// CHECK:STDOUT:     %c.patt: %Class = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %Class = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.param: %Class = value_param runtime_param0
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnIncomplete.decl: %ReturnIncomplete.type = fn_decl @ReturnIncomplete [template = constants.%ReturnIncomplete] {
// CHECK:STDOUT:     %return.patt: %Class = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Class = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %return.param: ref %Class = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTakeIncomplete.decl: %CallTakeIncomplete.type = fn_decl @CallTakeIncomplete [template = constants.%CallTakeIncomplete] {
// CHECK:STDOUT:     %p.patt: %.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc92: type = ptr_type %Class [template = constants.%.3]
// CHECK:STDOUT:     %p.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.3 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallReturnIncomplete.decl: %CallReturnIncomplete.type = fn_decl @CallReturnIncomplete [template = constants.%CallReturnIncomplete] {} {}
// CHECK:STDOUT:   %IncompleteAddrSelf.decl: type = class_decl @IncompleteAddrSelf [template = constants.%IncompleteAddrSelf] {} {}
// CHECK:STDOUT:   %CallIncompleteAddrSelf.decl: %CallIncompleteAddrSelf.type = fn_decl @CallIncompleteAddrSelf [template = constants.%CallIncompleteAddrSelf] {
// CHECK:STDOUT:     %p.patt: %.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc136: type = ptr_type %Class [template = constants.%.3]
// CHECK:STDOUT:     %p.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.3 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: class @IncompleteAddrSelf {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %.3 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.3 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc133_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc133_24: type = ptr_type %Class [template = constants.%.3]
// CHECK:STDOUT:     %self.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.3 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc134: <witness> = complete_type_witness %.2 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%IncompleteAddrSelf
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallClassFunction() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %Function.ref: <error> = name_ref Function, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertFromStruct() -> %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc42_43: %.2 = struct_literal ()
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %.3) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc52: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberAccess(%p.param_patt: %.3) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc63: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Copy(%p.param_patt: %.3) -> %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc74: ref %Class = deref %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let(%p.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc85: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc85: ref %Class = deref %p.ref
// CHECK:STDOUT:   %c: <error> = bind_name c, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeIncomplete(%c.param_patt: %Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnIncomplete() -> %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTakeIncomplete(%p.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeIncomplete.ref.loc103: %TakeIncomplete.type = name_ref TakeIncomplete, file.%TakeIncomplete.decl [template = constants.%TakeIncomplete]
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc103: ref %Class = deref %p.ref
// CHECK:STDOUT:   %TakeIncomplete.call.loc103: init %empty_tuple.type = call %TakeIncomplete.ref.loc103(<error>)
// CHECK:STDOUT:   %TakeIncomplete.ref.loc115: %TakeIncomplete.type = name_ref TakeIncomplete, file.%TakeIncomplete.decl [template = constants.%TakeIncomplete]
// CHECK:STDOUT:   %.loc115: %.2 = struct_literal ()
// CHECK:STDOUT:   %TakeIncomplete.call.loc115: init %empty_tuple.type = call %TakeIncomplete.ref.loc115(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallReturnIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnIncomplete.ref: %ReturnIncomplete.type = name_ref ReturnIncomplete, file.%ReturnIncomplete.decl [template = constants.%ReturnIncomplete]
// CHECK:STDOUT:   %ReturnIncomplete.call: init <error> = call %ReturnIncomplete.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.param_patt: %.3]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIncompleteAddrSelf(%p.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, %p
// CHECK:STDOUT:   %IncompleteAddrSelf.ref: type = name_ref IncompleteAddrSelf, file.%IncompleteAddrSelf.decl [template = constants.%IncompleteAddrSelf]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @IncompleteAddrSelf.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc148_4.1: ref %Class = deref %p.ref
// CHECK:STDOUT:   %.loc148_4.2: <bound method> = bound_method %.loc148_4.1, %F.ref
// CHECK:STDOUT:   %.1: <error> = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc148_4.2(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc11: <error> = field_decl c, element0 [template]
// CHECK:STDOUT:   %.loc12: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .c = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
