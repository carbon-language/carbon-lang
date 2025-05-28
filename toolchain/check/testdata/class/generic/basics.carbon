// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/basics.carbon

// --- forward_decl.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class C[]();
//@dump-sem-ir-end

// --- class.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class(T:! type) {
  fn GetAddr[addr self: Self*]() -> T* {
    return &self->k;
  }

  // TODO: Should this work? T is not necessarily copyable.
  fn GetValue[self: Self]() -> T {
    return self.k;
  }

  var k: T;
}

class Declaration(T:! type);
//@dump-sem-ir-end

// --- self_type.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  // `Self` is the same as `Class(T)` here.
  // TODO: Find a better way to test two types are the same.
  fn MakeSelf() -> Self { return {}; }
  fn MakeClass() -> Class(T) { return {}; }

  fn F() {
    //@dump-sem-ir-begin
    var c: Class(T) = MakeSelf();
    var s: Self = MakeClass();
    //@dump-sem-ir-end
  }
}

// --- field.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  var x: T;
}

fn F(c: Class(i32)) -> i32 {
  //@dump-sem-ir-begin
  return c.x;
  //@dump-sem-ir-end
}

fn G(T:! type, c: Class(T)) -> T {
  //@dump-sem-ir-begin
  return c.x;
  //@dump-sem-ir-end
}

fn H(U:! type, c: Class(U)) -> U {
  //@dump-sem-ir-begin
  return c.x;
  //@dump-sem-ir-end
}

// --- fail_derived_to_base.carbon

library "[[@TEST_NAME]]";

fn Int(N: Core.IntLiteral()) -> type = "int.make_type_signed";

base class B {}

class A(N:! i32) {
  extend base: B;

  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:10: error: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   var n: Int(N);
  // CHECK:STDERR:          ^~~~~~
  var n: Int(N);
}

fn F(a: A(0)*) {
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:15: note: in `A(0)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   let b: B* = a;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  let b: B* = a;
}

// --- fail_self_param.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_self_param.carbon:[[@LINE+4]]:9: error: `self` parameter only allowed on functions [SelfParameterNotAllowed]
// CHECK:STDERR: class C[self:! type](x:! self) {}
// CHECK:STDERR:         ^~~~~~~~~~~
// CHECK:STDERR:
class C[self:! type](x:! self) {}
var v: C(0);

// CHECK:STDOUT: --- forward_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetAddr: %GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetValue: %GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.2ae: <witness> = require_complete_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Declaration.type: type = generic_class_type @Declaration [concrete]
// CHECK:STDOUT:   %Declaration.generic: %Declaration.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [concrete = constants.%Declaration.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc18_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc18_19.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T.loc5_13.2) [symbolic = %GetAddr.type (constants.%GetAddr.type)]
// CHECK:STDOUT:   %GetAddr: @Class.%GetAddr.type (%GetAddr.type) = struct_value () [symbolic = %GetAddr (constants.%GetAddr)]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T.loc5_13.2) [symbolic = %GetValue.type (constants.%GetValue.type)]
// CHECK:STDOUT:   %GetValue: @Class.%GetValue.type (%GetValue.type) = struct_value () [symbolic = %GetValue (constants.%GetValue)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_13.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_13.2) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc5_13.2 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.k.loc16_1.2: type = struct_type {.k: @Class.%T.loc5_13.2 (%T)} [symbolic = %struct_type.k.loc16_1.2 (constants.%struct_type.k)]
// CHECK:STDOUT:   %complete_type.loc16_1.2: <witness> = complete_type_witness %struct_type.k.loc16_1.2 [symbolic = %complete_type.loc16_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type) = fn_decl @GetAddr [symbolic = @Class.%GetAddr (constants.%GetAddr)] {
// CHECK:STDOUT:       %self.patt: @GetAddr.%pattern_type.loc6_19 (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @GetAddr.%pattern_type.loc6_19 (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:       %return.patt: @GetAddr.%pattern_type.loc6_34 (%pattern_type.afe) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @GetAddr.%pattern_type.loc6_34 (%pattern_type.afe) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc5_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_38.2: type = ptr_type %T.ref [symbolic = %ptr.loc6_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:       %self.param: @GetAddr.%ptr.loc6_29.1 (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc6_29: type = splice_block %ptr.loc6_29.2 [symbolic = %ptr.loc6_29.1 (constants.%ptr.955)] {
// CHECK:STDOUT:         %.loc6_25: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_25 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %ptr.loc6_29.2: type = ptr_type %Self.ref [symbolic = %ptr.loc6_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GetAddr.%ptr.loc6_29.1 (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetAddr.%ptr.loc6_38.1 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetAddr.%ptr.loc6_38.1 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetValue.decl: @Class.%GetValue.type (%GetValue.type) = fn_decl @GetValue [symbolic = @Class.%GetValue (constants.%GetValue)] {
// CHECK:STDOUT:       %self.patt: @GetValue.%pattern_type.loc11_15 (%pattern_type.3c1) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @GetValue.%pattern_type.loc11_15 (%pattern_type.3c1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @GetValue.%pattern_type.loc11_29 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @GetValue.%pattern_type.loc11_29 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc5_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @GetValue.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc11_21.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc11_21.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_21.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GetValue.%Class (%Class) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetValue.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetValue.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_13.1 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc15: @Class.%Class.elem (%Class.elem) = field_decl k, element0 [concrete]
// CHECK:STDOUT:     %struct_type.k.loc16_1.1: type = struct_type {.k: %T} [symbolic = %struct_type.k.loc16_1.2 (constants.%struct_type.k)]
// CHECK:STDOUT:     %complete_type.loc16_1.1: <witness> = complete_type_witness %struct_type.k.loc16_1.1 [symbolic = %complete_type.loc16_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc16_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:     .GetValue = %GetValue.decl
// CHECK:STDOUT:     .k = %.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Declaration(%T.loc18_19.1: type) {
// CHECK:STDOUT:   %T.loc18_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc18_19.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetAddr(@Class.%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr.loc6_29.1: type = ptr_type %Class [symbolic = %ptr.loc6_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %ptr.loc6_29.1 [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.9e0)]
// CHECK:STDOUT:   %ptr.loc6_38.1: type = ptr_type %T [symbolic = %ptr.loc6_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type.loc6_34: type = pattern_type %ptr.loc6_38.1 [symbolic = %pattern_type.loc6_34 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_34: <witness> = require_complete_type %ptr.loc6_38.1 [symbolic = %require_complete.loc6_34 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type %ptr.loc6_29.1 [symbolic = %require_complete.loc6_23 (constants.%require_complete.2ae)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %Class [symbolic = %require_complete.loc7 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @GetAddr.%ptr.loc6_29.1 (%ptr.955)) -> @GetAddr.%ptr.loc6_38.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%ptr.loc6_29.1 (%ptr.955) = name_ref self, %self
// CHECK:STDOUT:     %.loc7_17.1: ref @GetAddr.%Class (%Class) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @GetAddr.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc15 [concrete = @Class.%.loc15]
// CHECK:STDOUT:     %.loc7_17.2: ref @GetAddr.%T (%T) = class_element_access %.loc7_17.1, element0
// CHECK:STDOUT:     %addr: @GetAddr.%ptr.loc6_38.1 (%ptr.79f) = addr_of %.loc7_17.2
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetValue(@Class.%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc11_15: type = pattern_type %Class [symbolic = %pattern_type.loc11_15 (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %pattern_type.loc11_29: type = pattern_type %T [symbolic = %pattern_type.loc11_29 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %Class [symbolic = %require_complete.loc11 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %T [symbolic = %require_complete.loc12 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @GetValue.%Class (%Class)) -> @GetValue.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetValue.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @GetValue.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc15 [concrete = @Class.%.loc15]
// CHECK:STDOUT:     %.loc12_16.1: ref @GetValue.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc12_16.2: @GetValue.%T (%T) = bind_value %.loc12_16.1
// CHECK:STDOUT:     return %.loc12_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr
// CHECK:STDOUT:   %GetValue.type => constants.%GetValue.type
// CHECK:STDOUT:   %GetValue => constants.%GetValue
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.k.loc16_1.2 => constants.%struct_type.k
// CHECK:STDOUT:   %complete_type.loc16_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr.loc6_29.1 => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.9e0
// CHECK:STDOUT:   %ptr.loc6_38.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type.loc6_34 => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetValue(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc11_15 => constants.%pattern_type.3c1
// CHECK:STDOUT:   %pattern_type.loc11_29 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T) {
// CHECK:STDOUT:   %T.loc18_19.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeSelf: %MakeSelf.type = struct_value () [symbolic]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeClass: %MakeClass.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %MakeSelf.specific_fn: <specific function> = specific_function %MakeSelf, @MakeSelf(%T) [symbolic]
// CHECK:STDOUT:   %MakeClass.specific_fn: <specific function> = specific_function %MakeClass, @MakeClass(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .MakeSelf = %MakeSelf.decl
// CHECK:STDOUT:     .Class = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .MakeClass = %MakeClass.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc12_19.2: type = class_type @Class, @Class(%T) [symbolic = %Class.loc12_19.2 (constants.%Class)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc12_19.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc12_19.2 [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic = %MakeSelf.type (constants.%MakeSelf.type)]
// CHECK:STDOUT:   %MakeSelf: @F.%MakeSelf.type (%MakeSelf.type) = struct_value () [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:   %MakeSelf.specific_fn.loc12_23.2: <specific function> = specific_function %MakeSelf, @MakeSelf(%T) [symbolic = %MakeSelf.specific_fn.loc12_23.2 (constants.%MakeSelf.specific_fn)]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic = %MakeClass.type (constants.%MakeClass.type)]
// CHECK:STDOUT:   %MakeClass: @F.%MakeClass.type (%MakeClass.type) = struct_value () [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:   %MakeClass.specific_fn.loc13_19.2: <specific function> = specific_function %MakeClass, @MakeClass(%T) [symbolic = %MakeClass.specific_fn.loc13_19.2 (constants.%MakeClass.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: @F.%pattern_type (%pattern_type.3c1) = binding_pattern c [concrete]
// CHECK:STDOUT:       %c.var_patt: @F.%pattern_type (%pattern_type.3c1) = var_pattern %c.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.var: ref @F.%Class.loc12_19.2 (%Class) = var %c.var_patt
// CHECK:STDOUT:     %.loc12_23: @F.%MakeSelf.type (%MakeSelf.type) = specific_constant @Class.%MakeSelf.decl, @Class(constants.%T) [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.ref: @F.%MakeSelf.type (%MakeSelf.type) = name_ref MakeSelf, %.loc12_23 [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.specific_fn.loc12_23.1: <specific function> = specific_function %MakeSelf.ref, @MakeSelf(constants.%T) [symbolic = %MakeSelf.specific_fn.loc12_23.2 (constants.%MakeSelf.specific_fn)]
// CHECK:STDOUT:     %.loc12_5: ref @F.%Class.loc12_19.2 (%Class) = splice_block %c.var {}
// CHECK:STDOUT:     %MakeSelf.call: init @F.%Class.loc12_19.2 (%Class) = call %MakeSelf.specific_fn.loc12_23.1() to %.loc12_5
// CHECK:STDOUT:     assign %c.var, %MakeSelf.call
// CHECK:STDOUT:     %.loc12_19: type = splice_block %Class.loc12_19.1 [symbolic = %Class.loc12_19.2 (constants.%Class)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc12_19.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc12_19.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: ref @F.%Class.loc12_19.2 (%Class) = bind_name c, %c.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %s.patt: @F.%pattern_type (%pattern_type.3c1) = binding_pattern s [concrete]
// CHECK:STDOUT:       %s.var_patt: @F.%pattern_type (%pattern_type.3c1) = var_pattern %s.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s.var: ref @F.%Class.loc12_19.2 (%Class) = var %s.var_patt
// CHECK:STDOUT:     %.loc13_19: @F.%MakeClass.type (%MakeClass.type) = specific_constant @Class.%MakeClass.decl, @Class(constants.%T) [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %MakeClass.ref: @F.%MakeClass.type (%MakeClass.type) = name_ref MakeClass, %.loc13_19 [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %MakeClass.specific_fn.loc13_19.1: <specific function> = specific_function %MakeClass.ref, @MakeClass(constants.%T) [symbolic = %MakeClass.specific_fn.loc13_19.2 (constants.%MakeClass.specific_fn)]
// CHECK:STDOUT:     %.loc13_5: ref @F.%Class.loc12_19.2 (%Class) = splice_block %s.var {}
// CHECK:STDOUT:     %MakeClass.call: init @F.%Class.loc12_19.2 (%Class) = call %MakeClass.specific_fn.loc13_19.1() to %.loc13_5
// CHECK:STDOUT:     assign %s.var, %MakeClass.call
// CHECK:STDOUT:     %.loc13_12.1: type = splice_block %Self.ref [symbolic = %Class.loc12_19.2 (constants.%Class)] {
// CHECK:STDOUT:       %.loc13_12.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class.loc12_19.2 (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc13_12.2 [symbolic = %Class.loc12_19.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: ref @F.%Class.loc12_19.2 (%Class) = bind_name s, %s.var
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MakeSelf.type => constants.%MakeSelf.type
// CHECK:STDOUT:   %MakeSelf => constants.%MakeSelf
// CHECK:STDOUT:   %MakeClass.type => constants.%MakeClass.type
// CHECK:STDOUT:   %MakeClass => constants.%MakeClass
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Class.fe1b2d.1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem.e262de.1: type = unbound_element_type %Class.fe1b2d.1, %T [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.247: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Class.elem.2d8: type = unbound_element_type %Class.247, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.3c18fb.1: type = pattern_type %Class.fe1b2d.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %Class.fe1b2d.2: type = class_type @Class, @Class(%U) [symbolic]
// CHECK:STDOUT:   %pattern_type.3c18fb.2: type = pattern_type %Class.fe1b2d.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %Class.elem.e262de.2: type = unbound_element_type %Class.fe1b2d.2, %U [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.247 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %Class.elem.2d8 = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc10_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc10_11.2: %i32 = bind_value %.loc10_11.1
// CHECK:STDOUT:   return %.loc10_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc14_6.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc14_26.2, %T.loc14_6.2 [symbolic = %Class.elem (constants.%Class.elem.e262de.1)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %T.loc14_6.2 [symbolic = %require_complete.loc16 (constants.%require_complete.4aeca8.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @G.%Class.loc14_26.2 (%Class.fe1b2d.1)) -> @G.%T.loc14_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc14_26.2 (%Class.fe1b2d.1) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%Class.elem (%Class.elem.e262de.1) = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc16_11.1: ref @G.%T.loc14_6.2 (%T) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc16_11.2: @G.%T.loc14_6.2 (%T) = bind_value %.loc16_11.1
// CHECK:STDOUT:     return %.loc16_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc20_6.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc20_26.2, %U.loc20_6.2 [symbolic = %Class.elem (constants.%Class.elem.e262de.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @H.%Class.loc20_26.2 (%Class.fe1b2d.2)) -> @H.%U.loc20_6.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc20_26.2 (%Class.fe1b2d.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%Class.elem (%Class.elem.e262de.2) = name_ref x, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc22_11.1: ref @H.%U.loc20_6.2 (%U) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc22_11.2: @H.%U.loc20_6.2 (%U) = bind_value %.loc22_11.1
// CHECK:STDOUT:     return %.loc22_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc14_6.2 => constants.%T
// CHECK:STDOUT:   %Class.loc14_26.2 => constants.%Class.fe1b2d.1
// CHECK:STDOUT:   %pattern_type.loc14_16 => constants.%pattern_type.3c18fb.1
// CHECK:STDOUT:   %pattern_type.loc14_29 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U) {
// CHECK:STDOUT:   %U.loc20_6.2 => constants.%U
// CHECK:STDOUT:   %Class.loc20_26.2 => constants.%Class.fe1b2d.2
// CHECK:STDOUT:   %pattern_type.loc20_16 => constants.%pattern_type.3c18fb.2
// CHECK:STDOUT:   %pattern_type.loc20_29 => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
