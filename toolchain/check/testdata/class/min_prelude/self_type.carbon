// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/self_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/self_type.carbon

// --- self_type.carbon

library "[[@TEST_NAME]]";

class Class {
  //@dump-sem-ir-begin
  fn F[self: Self]() -> ();
  //@dump-sem-ir-end

  fn Make() -> Self {
    returned var s: Self;
    s = {.p = &s};
    return var;
  }
  var p: Self*;
}

//@dump-sem-ir-begin
fn Class.F[self: Self]() -> () {
  return (*self.p).F();
}
//@dump-sem-ir-end

// --- self_conversion.carbon

library "[[@TEST_NAME]]";

base class Base {
  var a: ();
}

//@dump-sem-ir-begin
class Derived {
  extend base: Base;

  fn SelfBase[self: Base]() -> ();
  fn AddrSelfBase[addr self: Base*]();
}
//@dump-sem-ir-end

fn Derived.SelfBase[self: Base]() -> () {
  //@dump-sem-ir-begin
  return self.a;
  //@dump-sem-ir-end
}

fn Derived.AddrSelfBase[addr self: Base*]() {
  //@dump-sem-ir-begin
  (*self).a = ();
  //@dump-sem-ir-end
}

fn Call(p: Derived*) -> () {
  (*p).AddrSelfBase();
  return (*p).SelfBase();
}

// --- fail_self_type_member.carbon

class C {}

class Class {
  var b: C;
}

fn F() -> {} {
  var c1: Class = {.b = {}};
  // CHECK:STDERR: fail_self_type_member.carbon:[[@LINE+8]]:17: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   var c2: Class.Self = c1;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_type_member.carbon:[[@LINE+4]]:17: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var c2: Class.Self = c1;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  var c2: Class.Self = c1;
  return c2.b;
}

// CHECK:STDOUT: --- self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %ptr.e71 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_30.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_30.2: type = converted %.loc18_30.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param.loc18: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc18: %Class = bind_name self, %self.param.loc18
// CHECK:STDOUT:     %return.param.loc18: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return.loc18: ref %empty_tuple.type = return_slot %return.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param.loc6: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc6: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc6: %Class = bind_name self, %self.param.loc6
// CHECK:STDOUT:     %return.param.loc6: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return.loc6: ref %empty_tuple.type = return_slot %return.param.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .p = %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param.loc18: %Class) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc18
// CHECK:STDOUT:   %p.ref: %Class.elem = name_ref p, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:   %.loc19_16.1: ref %ptr.e71 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc19_16.2: %ptr.e71 = bind_value %.loc19_16.1
// CHECK:STDOUT:   %.loc19_11.1: ref %Class = deref %.loc19_16.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc19_11.1, %F.ref
// CHECK:STDOUT:   %.loc19_11.2: %Class = bind_value %.loc19_11.1
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%.loc19_11.2)
// CHECK:STDOUT:   %.loc19_22.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc19_22.2: ref %empty_tuple.type = temporary %.loc19_22.1, %F.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_23: %empty_tuple.type = converted %F.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc19_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %SelfBase.type: type = fn_type @SelfBase [concrete]
// CHECK:STDOUT:   %SelfBase: %SelfBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %AddrSelfBase.type: type = fn_type @AddrSelfBase [concrete]
// CHECK:STDOUT:   %AddrSelfBase: %AddrSelfBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.b1e: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base.b1e [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc10: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %SelfBase.decl: %SelfBase.type = fn_decl @SelfBase [concrete = constants.%SelfBase] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_33.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_33.2: type = converted %.loc12_33.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param.loc12: %Base = value_param call_param0
// CHECK:STDOUT:     %Base.ref.loc12: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc12: %Base = bind_name self, %self.param.loc12
// CHECK:STDOUT:     %return.param.loc12: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return.loc12: ref %empty_tuple.type = return_slot %return.param.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [concrete = constants.%AddrSelfBase] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc13: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %ptr.loc13 [concrete = constants.%ptr.11f] {
// CHECK:STDOUT:       %Base.ref.loc13: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:       %ptr.loc13: type = ptr_type %Base.ref.loc13 [concrete = constants.%ptr.11f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc13: %ptr.11f = bind_name self, %self.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete = constants.%struct_type.base.b1e]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc10
// CHECK:STDOUT:   .SelfBase = %SelfBase.decl
// CHECK:STDOUT:   .AddrSelfBase = %AddrSelfBase.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfBase(%self.param.loc17: %Base) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Base = name_ref self, %self.loc17
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc5_8 [concrete = @Base.%.loc5_8]
// CHECK:STDOUT:   %.loc19_14.1: ref %empty_tuple.type = class_element_access %self.ref, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_14.2: %empty_tuple.type = converted %.loc19_14.1, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc19_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrSelfBase(%self.param.loc23: %ptr.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.11f = name_ref self, %self.loc23
// CHECK:STDOUT:   %.loc25_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc5_8 [concrete = @Base.%.loc5_8]
// CHECK:STDOUT:   %.loc25_10: ref %empty_tuple.type = class_element_access %.loc25_4, element0
// CHECK:STDOUT:   %.loc25_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc25_16.2: init %empty_tuple.type = tuple_init () to %.loc25_10 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_13: init %empty_tuple.type = converted %.loc25_16.1, %.loc25_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc25_10, %.loc25_13
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
