// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/modifiers.carbon

// --- default_interfaces.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface DefaultFn {
  default fn F() {}
}

interface DefaultMethod {
  default fn M(unused self) {}
}

fn CallFn(generic d: DefaultFn) {
  d.F();
}

fn CallMethod[T: DefaultMethod](t: T) {
  t.M();
}
//@dump-sem-ir-end

// --- final_interfaces.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface FinalFn {
  final fn F() {}
}

interface FinalMethod {
  final fn M(unused self) {}
}

fn CallFn(generic d: FinalFn) {
  d.F();
}

fn CallMethod[T: FinalMethod](t: T) {
  t.M();
}
//@dump-sem-ir-end

// --- default_function.carbon

library "[[@TEST_NAME]]";
import library "default_interfaces";

//@dump-sem-ir-begin
class C {
  impl as DefaultFn {}
}

fn F() {
  CallFn(C);
}
//@dump-sem-ir-end

// --- default_method.carbon

library "[[@TEST_NAME]]";
import library "default_interfaces";

//@dump-sem-ir-begin
class C {
  impl as DefaultMethod {}
}

fn F() {
  let c: C = {};
  CallMethod(c);
}
//@dump-sem-ir-end

// --- default_function_overridden.carbon

library "[[@TEST_NAME]]";
import library "default_interfaces";

//@dump-sem-ir-begin
class C {
  fn G() { }
  impl as DefaultFn {
    fn F() { C.G(); }
  }
}

fn F() {
  CallFn(C);
}
//@dump-sem-ir-end

// --- default_method_overridden.carbon

library "[[@TEST_NAME]]";
import library "default_interfaces";

//@dump-sem-ir-begin
class C {
  fn G(unused self) { }
  impl as DefaultMethod {
    fn M(self) { self.G(); }
  }
}

fn F() {
  let c: C = {};
  CallMethod(c);
}
//@dump-sem-ir-end

// --- final_function.carbon

library "[[@TEST_NAME]]";
import library "final_interfaces";

//@dump-sem-ir-begin
class C {
  impl as FinalFn {}
}

fn F() {
  CallFn(C);
}
//@dump-sem-ir-end

// --- final_method.carbon

library "[[@TEST_NAME]]";
import library "final_interfaces";

//@dump-sem-ir-begin
class C {
  impl as FinalMethod {}
}

fn F() {
  let c: C = {};
  CallMethod(c);
}
//@dump-sem-ir-end

// --- todo_fail_inline_definition_without_modifier.carbon

interface I {
  // TODO: diagnose non-default or non-final interface members with definitions.
  fn F() {}
}

// --- fail_todo_deferred_modifiers.carbon

library "[[@TEST_NAME]]";

interface Modifiers {
  // CHECK:STDERR: fail_todo_deferred_modifiers.carbon:[[@LINE+2]]:3-9: error: TODO: `default` modifier currently requires an inline definition [ModifierFinalRequiresDefaultImpl]
  // CHECK:STDERR:
  default fn DefaultFunction();
  // CHECK:STDERR: fail_todo_deferred_modifiers.carbon:[[@LINE+2]]:3-9: error: TODO: `default` modifier currently requires an inline definition [ModifierFinalRequiresDefaultImpl]
  // CHECK:STDERR:
  default fn DefaultMethod(self);

  // CHECK:STDERR: fail_todo_deferred_modifiers.carbon:[[@LINE+2]]:3-7: error: TODO: `final` modifier currently requires an inline definition [ModifierFinalRequiresDefaultImpl]
  // CHECK:STDERR:
  final fn FinalFunction();
  // CHECK:STDERR: fail_todo_deferred_modifiers.carbon:[[@LINE+2]]:3-7: error: TODO: `final` modifier currently requires an inline definition [ModifierFinalRequiresDefaultImpl]
  // CHECK:STDERR:
  final fn FinalMethod(self);
}

// --- fail_impl_final.carbon

library "[[@TEST_NAME]]";
import library "final_interfaces";

class C {
  // CHECK:STDERR: fail_impl_final.carbon:[[@LINE+4]]:3-23: error: attempted to implement `M`, which is final in interface `FinalMethod` [ImplFinalFunction]
  // CHECK:STDERR: imported from: fail_impl_final.carbon:[[@LINE-4]]:1 [InImport]
  // CHECK:STDERR: final_interfaces.carbon:10:3-27: note: associated function M declared here [AssociatedFunctionHere]
  // CHECK:STDERR:
  impl as FinalMethod {
    fn M(unused self) { }
  }
}

// --- fail_modified_associated_constants.carbon

library "[[@TEST_NAME]]";

interface C {
  // CHECK:STDERR: fail_modified_associated_constants.carbon:[[@LINE+2]]:3-9: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:
  default let T: type = C;
  // CHECK:STDERR: fail_modified_associated_constants.carbon:[[@LINE+2]]:3-7: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:
  final let U: type = C;
}

// CHECK:STDOUT: --- default_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DefaultFn.type: type = facet_type <@DefaultFn> [concrete]
// CHECK:STDOUT:   %Self.4f0: %DefaultFn.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.type.99d: type = fn_type @DefaultFn.WithSelf.F, @DefaultFn.WithSelf(%Self.4f0) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.71b: %DefaultFn.WithSelf.F.type.99d = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultFn.assoc_type: type = assoc_entity_type @DefaultFn [concrete]
// CHECK:STDOUT:   %assoc0.251: %DefaultFn.assoc_type = assoc_entity element0, @DefaultFn.WithSelf.%DefaultFn.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %DefaultMethod.type: type = facet_type <@DefaultMethod> [concrete]
// CHECK:STDOUT:   %Self.be0: %DefaultMethod.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.be0 [symbolic]
// CHECK:STDOUT:   %pattern_type.671: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.b05: %pattern_type.671 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.c7e: %pattern_type.671 = wrapper_binding_pattern self, %self.param_patt.b05 [symbolic]
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.type.5c3: type = fn_type @DefaultMethod.WithSelf.M, @DefaultMethod.WithSelf(%Self.be0) [symbolic]
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.ff1: %DefaultMethod.WithSelf.M.type.5c3 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultMethod.assoc_type: type = assoc_entity_type @DefaultMethod [concrete]
// CHECK:STDOUT:   %assoc0.a0f: %DefaultMethod.assoc_type = assoc_entity element0, @DefaultMethod.WithSelf.%DefaultMethod.WithSelf.M.decl [concrete]
// CHECK:STDOUT:   %require_complete.09c: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.8dc: type = pattern_type %DefaultFn.type [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.8dc = symbolic_binding_pattern d, 0 [symbolic]
// CHECK:STDOUT:   %d: %DefaultFn.type = symbolic_binding d, 0 [symbolic]
// CHECK:STDOUT:   %CallFn.type: type = fn_type @CallFn [concrete]
// CHECK:STDOUT:   %CallFn: %CallFn.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.type.0f7: type = fn_type @DefaultFn.WithSelf.F, @DefaultFn.WithSelf(%d) [symbolic]
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.6f2: %DefaultFn.WithSelf.F.type.0f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %d.as_type: type = facet_access_type %d [symbolic]
// CHECK:STDOUT:   %DefaultFn.lookup_impl_witness: <witness> = lookup_impl_witness %d, @DefaultFn [symbolic]
// CHECK:STDOUT:   %.2c3: type = fn_type_with_self_type %DefaultFn.WithSelf.F.type.0f7, %d [symbolic]
// CHECK:STDOUT:   %impl.elem0.f87: %.2c3 = impl_witness_access %DefaultFn.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ba2: <specific function> = specific_impl_function %impl.elem0.f87, @DefaultFn.WithSelf.F(%d) [symbolic]
// CHECK:STDOUT:   %pattern_type.336: type = pattern_type %DefaultMethod.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.336 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %DefaultMethod.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.70f: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.70f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.70f = wrapper_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5bd: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.type.0cf: type = fn_type @DefaultMethod.WithSelf.M, @DefaultMethod.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.091: %DefaultMethod.WithSelf.M.type.0cf = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultMethod.lookup_impl_witness: <witness> = lookup_impl_witness %T, @DefaultMethod [symbolic]
// CHECK:STDOUT:   %.506: type = fn_type_with_self_type %DefaultMethod.WithSelf.M.type.0cf, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0.f26: %.506 = impl_witness_access %DefaultMethod.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.382: %pattern_type.70f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.99e: %pattern_type.70f = wrapper_binding_pattern self, %self.param_patt.382 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.583: <specific function> = specific_impl_function %impl.elem0.f26, @DefaultMethod.WithSelf.M(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %DefaultFn.decl: type = interface_decl @DefaultFn [concrete = constants.%DefaultFn.type] {} {}
// CHECK:STDOUT:   %DefaultMethod.decl: type = interface_decl @DefaultMethod [concrete = constants.%DefaultMethod.type] {} {}
// CHECK:STDOUT:   %CallFn.decl: %CallFn.type = fn_decl @CallFn [concrete = constants.%CallFn] {
// CHECK:STDOUT:     %d.patt.loc13_20.1: %pattern_type.8dc = symbolic_binding_pattern d, 0 [symbolic = %d.patt.loc13_20.2 (constants.%d.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %DefaultFn.ref [concrete = constants.%DefaultFn.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %DefaultFn.ref: type = name_ref DefaultFn, file.%DefaultFn.decl [concrete = constants.%DefaultFn.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d.loc13_20.2: %DefaultFn.type = symbolic_binding d, 0 [symbolic = %d.loc13_20.1 (constants.%d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallMethod.decl: %CallMethod.type = fn_decl @CallMethod [concrete = constants.%CallMethod] {
// CHECK:STDOUT:     %T.patt.loc17_16.1: %pattern_type.336 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc17_34.1: @CallMethod.%pattern_type (%pattern_type.70f) = value_param_pattern [symbolic = %t.param_patt.loc17_34.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc17_34.1: @CallMethod.%pattern_type (%pattern_type.70f) = wrapper_binding_pattern t, %t.param_patt.loc17_34.1 [symbolic = %t.patt.loc17_34.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_18: type = splice_block %DefaultMethod.ref [concrete = constants.%DefaultMethod.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %DefaultMethod.ref: type = name_ref DefaultMethod, file.%DefaultMethod.decl [concrete = constants.%DefaultMethod.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_16.2: %DefaultMethod.type = symbolic_binding T, 0 [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @CallMethod.%T.as_type.loc17_36.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc17_36.1: type = splice_block %.loc17_36.2 [symbolic = %T.as_type.loc17_36.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %DefaultMethod.type = name_ref T, %T.loc17_16.2 [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc17_36.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc17_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc17_36.2: type = converted %T.ref, %T.as_type.loc17_36.2 [symbolic = %T.as_type.loc17_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @CallMethod.%T.as_type.loc17_36.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @DefaultFn {
// CHECK:STDOUT:   %Self: %DefaultFn.type = symbolic_binding Self, 0 [symbolic = constants.%Self.4f0]
// CHECK:STDOUT:   %DefaultFn.WithSelf.decl = interface_with_self_decl @DefaultFn [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.decl: @DefaultFn.WithSelf.%DefaultFn.WithSelf.F.type (%DefaultFn.WithSelf.F.type.99d) = fn_decl @DefaultFn.WithSelf.F [symbolic = @DefaultFn.WithSelf.%DefaultFn.WithSelf.F (constants.%DefaultFn.WithSelf.F.71b)] {} {}
// CHECK:STDOUT:   %assoc0: %DefaultFn.assoc_type = assoc_entity element0, %DefaultFn.WithSelf.F.decl [concrete = constants.%assoc0.251]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @DefaultFn.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@DefaultFn.WithSelf.%DefaultFn.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @DefaultMethod {
// CHECK:STDOUT:   %Self: %DefaultMethod.type = symbolic_binding Self, 0 [symbolic = constants.%Self.be0]
// CHECK:STDOUT:   %DefaultMethod.WithSelf.decl = interface_with_self_decl @DefaultMethod [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.decl: @DefaultMethod.WithSelf.%DefaultMethod.WithSelf.M.type (%DefaultMethod.WithSelf.M.type.5c3) = fn_decl @DefaultMethod.WithSelf.M [symbolic = @DefaultMethod.WithSelf.%DefaultMethod.WithSelf.M (constants.%DefaultMethod.WithSelf.M.ff1)] {
// CHECK:STDOUT:     %self.param_patt.loc10_23.1: @DefaultMethod.WithSelf.M.%pattern_type (%pattern_type.671) = value_param_pattern [symbolic = %self.param_patt.loc10_23.2 (constants.%self.param_patt.b05)]
// CHECK:STDOUT:     %self.patt.loc10_23.1: @DefaultMethod.WithSelf.M.%pattern_type (%pattern_type.671) = wrapper_binding_pattern self, %self.param_patt.loc10_23.1 [symbolic = %self.patt.loc10_23.2 (constants.%self.patt.c7e)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @DefaultMethod.WithSelf.M.%Self.as_type.loc10_23.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc10_23.1: type = splice_block %.loc10_23.2 [symbolic = %Self.as_type.loc10_23.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %DefaultMethod.type = name_ref Self, @DefaultMethod.%Self [symbolic = %Self (constants.%Self.be0)]
// CHECK:STDOUT:       %Self.as_type.loc10_23.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc10_23.2: type = converted %Self.ref, %Self.as_type.loc10_23.2 [symbolic = %Self.as_type.loc10_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @DefaultMethod.WithSelf.M.%Self.as_type.loc10_23.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %DefaultMethod.assoc_type = assoc_entity element0, %DefaultMethod.WithSelf.M.decl [concrete = constants.%assoc0.a0f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .M = @DefaultMethod.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@DefaultMethod.WithSelf.%DefaultMethod.WithSelf.M.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DefaultFn.WithSelf.F(@DefaultFn.%Self: %DefaultFn.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DefaultMethod.WithSelf.M(@DefaultMethod.%Self: %DefaultMethod.type) {
// CHECK:STDOUT:   %Self: %DefaultMethod.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.be0)]
// CHECK:STDOUT:   %Self.as_type.loc10_23.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc10_23.1 [symbolic = %pattern_type (constants.%pattern_type.671)]
// CHECK:STDOUT:   %self.param_patt.loc10_23.2: @DefaultMethod.WithSelf.M.%pattern_type (%pattern_type.671) = value_param_pattern [symbolic = %self.param_patt.loc10_23.2 (constants.%self.param_patt.b05)]
// CHECK:STDOUT:   %self.patt.loc10_23.2: @DefaultMethod.WithSelf.M.%pattern_type (%pattern_type.671) = wrapper_binding_pattern self, %self.param_patt.loc10_23.2 [symbolic = %self.patt.loc10_23.2 (constants.%self.patt.c7e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc10_23.1 [symbolic = %require_complete (constants.%require_complete.09c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @DefaultMethod.WithSelf.M.%Self.as_type.loc10_23.1 (%Self.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallFn(%d.loc13_20.2: %DefaultFn.type) {
// CHECK:STDOUT:   %d.patt.loc13_20.2: %pattern_type.8dc = symbolic_binding_pattern d, 0 [symbolic = %d.patt.loc13_20.2 (constants.%d.patt)]
// CHECK:STDOUT:   %d.loc13_20.1: %DefaultFn.type = symbolic_binding d, 0 [symbolic = %d.loc13_20.1 (constants.%d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %d.as_type.loc14_4.2: type = facet_access_type %d.loc13_20.1 [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.type: type = fn_type @DefaultFn.WithSelf.F, @DefaultFn.WithSelf(%d.loc13_20.1) [symbolic = %DefaultFn.WithSelf.F.type (constants.%DefaultFn.WithSelf.F.type.0f7)]
// CHECK:STDOUT:   %.loc14_4.2: type = fn_type_with_self_type %DefaultFn.WithSelf.F.type, %d.loc13_20.1 [symbolic = %.loc14_4.2 (constants.%.2c3)]
// CHECK:STDOUT:   %DefaultFn.lookup_impl_witness: <witness> = lookup_impl_witness %d.loc13_20.1, @DefaultFn [symbolic = %DefaultFn.lookup_impl_witness (constants.%DefaultFn.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @CallFn.%.loc14_4.2 (%.2c3) = impl_witness_access %DefaultFn.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.f87)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @DefaultFn.WithSelf.F(%d.loc13_20.1) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.ba2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %d.ref: %DefaultFn.type = name_ref d, %d.loc13_20.2 [symbolic = %d.loc13_20.1 (constants.%d)]
// CHECK:STDOUT:     %d.as_type.loc14_4.1: type = facet_access_type %d.ref [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:     %.loc14_4.1: type = converted %d.ref, %d.as_type.loc14_4.1 [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:     %F.ref: %DefaultFn.assoc_type = name_ref F, @DefaultFn.WithSelf.%assoc0 [concrete = constants.%assoc0.251]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @CallFn.%.loc14_4.2 (%.2c3) = impl_witness_access constants.%DefaultFn.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.f87)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @DefaultFn.WithSelf.F(constants.%d) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.ba2)]
// CHECK:STDOUT:     %DefaultFn.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallMethod(%T.loc17_16.2: %DefaultMethod.type) {
// CHECK:STDOUT:   %T.patt.loc17_16.2: %pattern_type.336 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc17_16.1: %DefaultMethod.type = symbolic_binding T, 0 [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc17_36.1: type = facet_access_type %T.loc17_16.1 [symbolic = %T.as_type.loc17_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc17_36.1 [symbolic = %pattern_type (constants.%pattern_type.70f)]
// CHECK:STDOUT:   %t.param_patt.loc17_34.2: @CallMethod.%pattern_type (%pattern_type.70f) = value_param_pattern [symbolic = %t.param_patt.loc17_34.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc17_34.2: @CallMethod.%pattern_type (%pattern_type.70f) = wrapper_binding_pattern t, %t.param_patt.loc17_34.2 [symbolic = %t.patt.loc17_34.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc17_36.1 [symbolic = %require_complete (constants.%require_complete.5bd)]
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.type: type = fn_type @DefaultMethod.WithSelf.M, @DefaultMethod.WithSelf(%T.loc17_16.1) [symbolic = %DefaultMethod.WithSelf.M.type (constants.%DefaultMethod.WithSelf.M.type.0cf)]
// CHECK:STDOUT:   %.loc18_4: type = fn_type_with_self_type %DefaultMethod.WithSelf.M.type, %T.loc17_16.1 [symbolic = %.loc18_4 (constants.%.506)]
// CHECK:STDOUT:   %DefaultMethod.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc17_16.1, @DefaultMethod [symbolic = %DefaultMethod.lookup_impl_witness (constants.%DefaultMethod.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc18_4.2: @CallMethod.%.loc18_4 (%.506) = impl_witness_access %DefaultMethod.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0.f26)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_4.2: <specific function> = specific_impl_function %impl.elem0.loc18_4.2, @DefaultMethod.WithSelf.M(%T.loc17_16.1) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.583)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @CallMethod.%T.as_type.loc17_36.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @CallMethod.%T.as_type.loc17_36.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %M.ref: %DefaultMethod.assoc_type = name_ref M, @DefaultMethod.WithSelf.%assoc0 [concrete = constants.%assoc0.a0f]
// CHECK:STDOUT:     %impl.elem0.loc18_4.1: @CallMethod.%.loc18_4 (%.506) = impl_witness_access constants.%DefaultMethod.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0.f26)]
// CHECK:STDOUT:     %bound_method.loc18_4: <bound method> = bound_method %t.ref, %impl.elem0.loc18_4.1
// CHECK:STDOUT:     %.loc18_7.1: %DefaultMethod.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:     %.loc18_7.2: %DefaultMethod.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc18_4.1: <specific function> = specific_impl_function %impl.elem0.loc18_4.1, @DefaultMethod.WithSelf.M(constants.%T) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.583)]
// CHECK:STDOUT:     %bound_method.loc18_7: <bound method> = bound_method %t.ref, %specific_impl_fn.loc18_4.1
// CHECK:STDOUT:     %DefaultMethod.WithSelf.M.call: init %empty_tuple.type = call %bound_method.loc18_7(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultFn.WithSelf(constants.%Self.4f0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.4f0
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.type => constants.%DefaultFn.WithSelf.F.type.99d
// CHECK:STDOUT:   %DefaultFn.WithSelf.F => constants.%DefaultFn.WithSelf.F.71b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultFn.WithSelf.F(constants.%Self.4f0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultMethod.WithSelf(constants.%Self.be0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.be0
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.type => constants.%DefaultMethod.WithSelf.M.type.5c3
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M => constants.%DefaultMethod.WithSelf.M.ff1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultMethod.WithSelf.M(constants.%Self.be0) {
// CHECK:STDOUT:   %Self => constants.%Self.be0
// CHECK:STDOUT:   %Self.as_type.loc10_23.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.671
// CHECK:STDOUT:   %self.param_patt.loc10_23.2 => constants.%self.param_patt.b05
// CHECK:STDOUT:   %self.patt.loc10_23.2 => constants.%self.patt.c7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallFn(constants.%d) {
// CHECK:STDOUT:   %d.patt.loc13_20.2 => constants.%d.patt
// CHECK:STDOUT:   %d.loc13_20.1 => constants.%d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultFn.WithSelf(constants.%d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%d
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.type => constants.%DefaultFn.WithSelf.F.type.0f7
// CHECK:STDOUT:   %DefaultFn.WithSelf.F => constants.%DefaultFn.WithSelf.F.6f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultFn.WithSelf.F(constants.%d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallMethod(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc17_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_16.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc17_36.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.70f
// CHECK:STDOUT:   %t.param_patt.loc17_34.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc17_34.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultMethod.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.type => constants.%DefaultMethod.WithSelf.M.type.0cf
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M => constants.%DefaultMethod.WithSelf.M.091
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DefaultMethod.WithSelf.M(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type.loc10_23.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.70f
// CHECK:STDOUT:   %self.param_patt.loc10_23.2 => constants.%self.param_patt.382
// CHECK:STDOUT:   %self.patt.loc10_23.2 => constants.%self.patt.99e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %FinalFn.type: type = facet_type <@FinalFn> [concrete]
// CHECK:STDOUT:   %Self.49e: %FinalFn.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %FinalFn.WithSelf.F.type.2c0: type = fn_type @FinalFn.WithSelf.F, @FinalFn.WithSelf(%Self.49e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %FinalFn.WithSelf.F.744: %FinalFn.WithSelf.F.type.2c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %FinalFn.assoc_type: type = assoc_entity_type @FinalFn [concrete]
// CHECK:STDOUT:   %assoc0.d6c: %FinalFn.assoc_type = assoc_entity element0, @FinalFn.WithSelf.%FinalFn.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %FinalMethod.type: type = facet_type <@FinalMethod> [concrete]
// CHECK:STDOUT:   %Self.e80: %FinalMethod.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e80 [symbolic]
// CHECK:STDOUT:   %pattern_type.32b: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.724: %pattern_type.32b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.537: %pattern_type.32b = wrapper_binding_pattern self, %self.param_patt.724 [symbolic]
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.type.efa: type = fn_type @FinalMethod.WithSelf.M, @FinalMethod.WithSelf(%Self.e80) [symbolic]
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.68b: %FinalMethod.WithSelf.M.type.efa = struct_value () [symbolic]
// CHECK:STDOUT:   %FinalMethod.assoc_type: type = assoc_entity_type @FinalMethod [concrete]
// CHECK:STDOUT:   %assoc0.e67: %FinalMethod.assoc_type = assoc_entity element0, @FinalMethod.WithSelf.%FinalMethod.WithSelf.M.decl [concrete]
// CHECK:STDOUT:   %require_complete.e87: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.778: type = pattern_type %FinalFn.type [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.778 = symbolic_binding_pattern d, 0 [symbolic]
// CHECK:STDOUT:   %d: %FinalFn.type = symbolic_binding d, 0 [symbolic]
// CHECK:STDOUT:   %CallFn.type: type = fn_type @CallFn [concrete]
// CHECK:STDOUT:   %CallFn: %CallFn.type = struct_value () [concrete]
// CHECK:STDOUT:   %FinalFn.WithSelf.F.type.94e: type = fn_type @FinalFn.WithSelf.F, @FinalFn.WithSelf(%d) [symbolic]
// CHECK:STDOUT:   %FinalFn.WithSelf.F.219: %FinalFn.WithSelf.F.type.94e = struct_value () [symbolic]
// CHECK:STDOUT:   %d.as_type: type = facet_access_type %d [symbolic]
// CHECK:STDOUT:   %FinalFn.lookup_impl_witness: <witness> = lookup_impl_witness %d, @FinalFn [symbolic]
// CHECK:STDOUT:   %.ec4: type = fn_type_with_self_type %FinalFn.WithSelf.F.type.94e, %d [symbolic]
// CHECK:STDOUT:   %impl.elem0.9ab: %.ec4 = impl_witness_access %FinalFn.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e4e: <specific function> = specific_impl_function %impl.elem0.9ab, @FinalFn.WithSelf.F(%d) [symbolic]
// CHECK:STDOUT:   %pattern_type.af4: type = pattern_type %FinalMethod.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.af4 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %FinalMethod.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.ab8: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.ab8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.ab8 = wrapper_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.3b2: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.type.913: type = fn_type @FinalMethod.WithSelf.M, @FinalMethod.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.4c6: %FinalMethod.WithSelf.M.type.913 = struct_value () [symbolic]
// CHECK:STDOUT:   %FinalMethod.lookup_impl_witness: <witness> = lookup_impl_witness %T, @FinalMethod [symbolic]
// CHECK:STDOUT:   %.e9f: type = fn_type_with_self_type %FinalMethod.WithSelf.M.type.913, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0.24d: %.e9f = impl_witness_access %FinalMethod.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.4d4: %pattern_type.ab8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.bc8: %pattern_type.ab8 = wrapper_binding_pattern self, %self.param_patt.4d4 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2e2: <specific function> = specific_impl_function %impl.elem0.24d, @FinalMethod.WithSelf.M(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %FinalFn.decl: type = interface_decl @FinalFn [concrete = constants.%FinalFn.type] {} {}
// CHECK:STDOUT:   %FinalMethod.decl: type = interface_decl @FinalMethod [concrete = constants.%FinalMethod.type] {} {}
// CHECK:STDOUT:   %CallFn.decl: %CallFn.type = fn_decl @CallFn [concrete = constants.%CallFn] {
// CHECK:STDOUT:     %d.patt.loc13_20.1: %pattern_type.778 = symbolic_binding_pattern d, 0 [symbolic = %d.patt.loc13_20.2 (constants.%d.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %FinalFn.ref [concrete = constants.%FinalFn.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %FinalFn.ref: type = name_ref FinalFn, file.%FinalFn.decl [concrete = constants.%FinalFn.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d.loc13_20.2: %FinalFn.type = symbolic_binding d, 0 [symbolic = %d.loc13_20.1 (constants.%d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallMethod.decl: %CallMethod.type = fn_decl @CallMethod [concrete = constants.%CallMethod] {
// CHECK:STDOUT:     %T.patt.loc17_16.1: %pattern_type.af4 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc17_32.1: @CallMethod.%pattern_type (%pattern_type.ab8) = value_param_pattern [symbolic = %t.param_patt.loc17_32.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc17_32.1: @CallMethod.%pattern_type (%pattern_type.ab8) = wrapper_binding_pattern t, %t.param_patt.loc17_32.1 [symbolic = %t.patt.loc17_32.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_18: type = splice_block %FinalMethod.ref [concrete = constants.%FinalMethod.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %FinalMethod.ref: type = name_ref FinalMethod, file.%FinalMethod.decl [concrete = constants.%FinalMethod.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_16.2: %FinalMethod.type = symbolic_binding T, 0 [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @CallMethod.%T.as_type.loc17_34.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc17_34.1: type = splice_block %.loc17_34.2 [symbolic = %T.as_type.loc17_34.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %FinalMethod.type = name_ref T, %T.loc17_16.2 [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc17_34.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc17_34.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc17_34.2: type = converted %T.ref, %T.as_type.loc17_34.2 [symbolic = %T.as_type.loc17_34.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @CallMethod.%T.as_type.loc17_34.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @FinalFn {
// CHECK:STDOUT:   %Self: %FinalFn.type = symbolic_binding Self, 0 [symbolic = constants.%Self.49e]
// CHECK:STDOUT:   %FinalFn.WithSelf.decl = interface_with_self_decl @FinalFn [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %FinalFn.WithSelf.F.decl: @FinalFn.WithSelf.%FinalFn.WithSelf.F.type (%FinalFn.WithSelf.F.type.2c0) = fn_decl @FinalFn.WithSelf.F [symbolic = @FinalFn.WithSelf.%FinalFn.WithSelf.F (constants.%FinalFn.WithSelf.F.744)] {} {}
// CHECK:STDOUT:   %assoc0: %FinalFn.assoc_type = assoc_entity element0, %FinalFn.WithSelf.F.decl [concrete = constants.%assoc0.d6c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @FinalFn.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@FinalFn.WithSelf.%FinalFn.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @FinalMethod {
// CHECK:STDOUT:   %Self: %FinalMethod.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e80]
// CHECK:STDOUT:   %FinalMethod.WithSelf.decl = interface_with_self_decl @FinalMethod [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.decl: @FinalMethod.WithSelf.%FinalMethod.WithSelf.M.type (%FinalMethod.WithSelf.M.type.efa) = fn_decl @FinalMethod.WithSelf.M [symbolic = @FinalMethod.WithSelf.%FinalMethod.WithSelf.M (constants.%FinalMethod.WithSelf.M.68b)] {
// CHECK:STDOUT:     %self.param_patt.loc10_21.1: @FinalMethod.WithSelf.M.%pattern_type (%pattern_type.32b) = value_param_pattern [symbolic = %self.param_patt.loc10_21.2 (constants.%self.param_patt.724)]
// CHECK:STDOUT:     %self.patt.loc10_21.1: @FinalMethod.WithSelf.M.%pattern_type (%pattern_type.32b) = wrapper_binding_pattern self, %self.param_patt.loc10_21.1 [symbolic = %self.patt.loc10_21.2 (constants.%self.patt.537)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @FinalMethod.WithSelf.M.%Self.as_type.loc10_21.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc10_21.1: type = splice_block %.loc10_21.2 [symbolic = %Self.as_type.loc10_21.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %FinalMethod.type = name_ref Self, @FinalMethod.%Self [symbolic = %Self (constants.%Self.e80)]
// CHECK:STDOUT:       %Self.as_type.loc10_21.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc10_21.2: type = converted %Self.ref, %Self.as_type.loc10_21.2 [symbolic = %Self.as_type.loc10_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @FinalMethod.WithSelf.M.%Self.as_type.loc10_21.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %FinalMethod.assoc_type = assoc_entity element0, %FinalMethod.WithSelf.M.decl [concrete = constants.%assoc0.e67]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .M = @FinalMethod.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@FinalMethod.WithSelf.%FinalMethod.WithSelf.M.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FinalFn.WithSelf.F(@FinalFn.%Self: %FinalFn.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FinalMethod.WithSelf.M(@FinalMethod.%Self: %FinalMethod.type) {
// CHECK:STDOUT:   %Self: %FinalMethod.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e80)]
// CHECK:STDOUT:   %Self.as_type.loc10_21.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc10_21.1 [symbolic = %pattern_type (constants.%pattern_type.32b)]
// CHECK:STDOUT:   %self.param_patt.loc10_21.2: @FinalMethod.WithSelf.M.%pattern_type (%pattern_type.32b) = value_param_pattern [symbolic = %self.param_patt.loc10_21.2 (constants.%self.param_patt.724)]
// CHECK:STDOUT:   %self.patt.loc10_21.2: @FinalMethod.WithSelf.M.%pattern_type (%pattern_type.32b) = wrapper_binding_pattern self, %self.param_patt.loc10_21.2 [symbolic = %self.patt.loc10_21.2 (constants.%self.patt.537)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc10_21.1 [symbolic = %require_complete (constants.%require_complete.e87)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @FinalMethod.WithSelf.M.%Self.as_type.loc10_21.1 (%Self.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallFn(%d.loc13_20.2: %FinalFn.type) {
// CHECK:STDOUT:   %d.patt.loc13_20.2: %pattern_type.778 = symbolic_binding_pattern d, 0 [symbolic = %d.patt.loc13_20.2 (constants.%d.patt)]
// CHECK:STDOUT:   %d.loc13_20.1: %FinalFn.type = symbolic_binding d, 0 [symbolic = %d.loc13_20.1 (constants.%d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %d.as_type.loc14_4.2: type = facet_access_type %d.loc13_20.1 [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:   %FinalFn.WithSelf.F.type: type = fn_type @FinalFn.WithSelf.F, @FinalFn.WithSelf(%d.loc13_20.1) [symbolic = %FinalFn.WithSelf.F.type (constants.%FinalFn.WithSelf.F.type.94e)]
// CHECK:STDOUT:   %.loc14_4.2: type = fn_type_with_self_type %FinalFn.WithSelf.F.type, %d.loc13_20.1 [symbolic = %.loc14_4.2 (constants.%.ec4)]
// CHECK:STDOUT:   %FinalFn.lookup_impl_witness: <witness> = lookup_impl_witness %d.loc13_20.1, @FinalFn [symbolic = %FinalFn.lookup_impl_witness (constants.%FinalFn.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @CallFn.%.loc14_4.2 (%.ec4) = impl_witness_access %FinalFn.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.9ab)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @FinalFn.WithSelf.F(%d.loc13_20.1) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.e4e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %d.ref: %FinalFn.type = name_ref d, %d.loc13_20.2 [symbolic = %d.loc13_20.1 (constants.%d)]
// CHECK:STDOUT:     %d.as_type.loc14_4.1: type = facet_access_type %d.ref [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:     %.loc14_4.1: type = converted %d.ref, %d.as_type.loc14_4.1 [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:     %F.ref: %FinalFn.assoc_type = name_ref F, @FinalFn.WithSelf.%assoc0 [concrete = constants.%assoc0.d6c]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @CallFn.%.loc14_4.2 (%.ec4) = impl_witness_access constants.%FinalFn.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.9ab)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @FinalFn.WithSelf.F(constants.%d) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.e4e)]
// CHECK:STDOUT:     %FinalFn.WithSelf.F.call: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallMethod(%T.loc17_16.2: %FinalMethod.type) {
// CHECK:STDOUT:   %T.patt.loc17_16.2: %pattern_type.af4 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc17_16.1: %FinalMethod.type = symbolic_binding T, 0 [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc17_34.1: type = facet_access_type %T.loc17_16.1 [symbolic = %T.as_type.loc17_34.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc17_34.1 [symbolic = %pattern_type (constants.%pattern_type.ab8)]
// CHECK:STDOUT:   %t.param_patt.loc17_32.2: @CallMethod.%pattern_type (%pattern_type.ab8) = value_param_pattern [symbolic = %t.param_patt.loc17_32.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc17_32.2: @CallMethod.%pattern_type (%pattern_type.ab8) = wrapper_binding_pattern t, %t.param_patt.loc17_32.2 [symbolic = %t.patt.loc17_32.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc17_34.1 [symbolic = %require_complete (constants.%require_complete.3b2)]
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.type: type = fn_type @FinalMethod.WithSelf.M, @FinalMethod.WithSelf(%T.loc17_16.1) [symbolic = %FinalMethod.WithSelf.M.type (constants.%FinalMethod.WithSelf.M.type.913)]
// CHECK:STDOUT:   %.loc18_4: type = fn_type_with_self_type %FinalMethod.WithSelf.M.type, %T.loc17_16.1 [symbolic = %.loc18_4 (constants.%.e9f)]
// CHECK:STDOUT:   %FinalMethod.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc17_16.1, @FinalMethod [symbolic = %FinalMethod.lookup_impl_witness (constants.%FinalMethod.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc18_4.2: @CallMethod.%.loc18_4 (%.e9f) = impl_witness_access %FinalMethod.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0.24d)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_4.2: <specific function> = specific_impl_function %impl.elem0.loc18_4.2, @FinalMethod.WithSelf.M(%T.loc17_16.1) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.2e2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @CallMethod.%T.as_type.loc17_34.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @CallMethod.%T.as_type.loc17_34.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %M.ref: %FinalMethod.assoc_type = name_ref M, @FinalMethod.WithSelf.%assoc0 [concrete = constants.%assoc0.e67]
// CHECK:STDOUT:     %impl.elem0.loc18_4.1: @CallMethod.%.loc18_4 (%.e9f) = impl_witness_access constants.%FinalMethod.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0.24d)]
// CHECK:STDOUT:     %bound_method.loc18_4: <bound method> = bound_method %t.ref, %impl.elem0.loc18_4.1
// CHECK:STDOUT:     %.loc18_7.1: %FinalMethod.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:     %.loc18_7.2: %FinalMethod.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc17_16.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc18_4.1: <specific function> = specific_impl_function %impl.elem0.loc18_4.1, @FinalMethod.WithSelf.M(constants.%T) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.2e2)]
// CHECK:STDOUT:     %bound_method.loc18_7: <bound method> = bound_method %t.ref, %specific_impl_fn.loc18_4.1
// CHECK:STDOUT:     %FinalMethod.WithSelf.M.call: init %empty_tuple.type = call %bound_method.loc18_7(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FinalFn.WithSelf(constants.%Self.49e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.49e
// CHECK:STDOUT:   %FinalFn.WithSelf.F.type => constants.%FinalFn.WithSelf.F.type.2c0
// CHECK:STDOUT:   %FinalFn.WithSelf.F => constants.%FinalFn.WithSelf.F.744
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FinalFn.WithSelf.F(constants.%Self.49e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FinalMethod.WithSelf(constants.%Self.e80) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.e80
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.type => constants.%FinalMethod.WithSelf.M.type.efa
// CHECK:STDOUT:   %FinalMethod.WithSelf.M => constants.%FinalMethod.WithSelf.M.68b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FinalMethod.WithSelf.M(constants.%Self.e80) {
// CHECK:STDOUT:   %Self => constants.%Self.e80
// CHECK:STDOUT:   %Self.as_type.loc10_21.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.32b
// CHECK:STDOUT:   %self.param_patt.loc10_21.2 => constants.%self.param_patt.724
// CHECK:STDOUT:   %self.patt.loc10_21.2 => constants.%self.patt.537
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallFn(constants.%d) {
// CHECK:STDOUT:   %d.patt.loc13_20.2 => constants.%d.patt
// CHECK:STDOUT:   %d.loc13_20.1 => constants.%d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FinalFn.WithSelf(constants.%d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%d
// CHECK:STDOUT:   %FinalFn.WithSelf.F.type => constants.%FinalFn.WithSelf.F.type.94e
// CHECK:STDOUT:   %FinalFn.WithSelf.F => constants.%FinalFn.WithSelf.F.219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FinalFn.WithSelf.F(constants.%d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallMethod(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc17_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_16.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc17_34.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ab8
// CHECK:STDOUT:   %t.param_patt.loc17_32.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc17_32.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FinalMethod.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.type => constants.%FinalMethod.WithSelf.M.type.913
// CHECK:STDOUT:   %FinalMethod.WithSelf.M => constants.%FinalMethod.WithSelf.M.4c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FinalMethod.WithSelf.M(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type.loc10_21.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ab8
// CHECK:STDOUT:   %self.param_patt.loc10_21.2 => constants.%self.param_patt.4d4
// CHECK:STDOUT:   %self.patt.loc10_21.2 => constants.%self.patt.bc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %DefaultFn.type: type = facet_type <@DefaultFn> [concrete]
// CHECK:STDOUT:   %.1d8: <witness> = impl_self_witness %C, @DefaultFn [concrete]
// CHECK:STDOUT:   %DefaultFn.impl_witness: <witness> = impl_witness @C.as.DefaultFn.impl.%DefaultFn.impl_witness_table [concrete]
// CHECK:STDOUT:   %DefaultFn.facet: %DefaultFn.type = facet_value %C, (%DefaultFn.impl_witness) [concrete]
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.type.08a: type = fn_type @DefaultFn.WithSelf.F, @DefaultFn.WithSelf(%DefaultFn.facet) [concrete]
// CHECK:STDOUT:   %DefaultFn.WithSelf.F.7e6: %DefaultFn.WithSelf.F.type.08a = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFn.type: type = fn_type @CallFn [concrete]
// CHECK:STDOUT:   %CallFn: %CallFn.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFn.specific_fn: <specific function> = specific_function %CallFn, @CallFn(%DefaultFn.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.DefaultFn: type = import_ref Main//default_interfaces, DefaultFn, loaded [concrete = constants.%DefaultFn.type]
// CHECK:STDOUT:   %Main.CallFn: %CallFn.type = import_ref Main//default_interfaces, CallFn, loaded [concrete = constants.%CallFn]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.DefaultFn.impl: %Self.ref as %DefaultFn.ref {
// CHECK:STDOUT:   %DefaultFn.impl_witness_table = impl_witness_table (constants.%DefaultFn.WithSelf.F.7e6), @C.as.DefaultFn.impl [concrete]
// CHECK:STDOUT:   %DefaultFn.impl_witness: <witness> = impl_witness %DefaultFn.impl_witness_table [concrete = constants.%DefaultFn.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %DefaultFn.ref
// CHECK:STDOUT:   witness = %DefaultFn.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.DefaultFn.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %DefaultFn.ref: type = name_ref DefaultFn, imports.%Main.DefaultFn [concrete = constants.%DefaultFn.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.DefaultFn.impl.%Self.ref, @DefaultFn [concrete = constants.%.1d8]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .DefaultFn = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallFn.ref: %CallFn.type = name_ref CallFn, imports.%Main.CallFn [concrete = constants.%CallFn]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %DefaultFn.facet: %DefaultFn.type = facet_value %C.ref, (constants.%DefaultFn.impl_witness) [concrete = constants.%DefaultFn.facet]
// CHECK:STDOUT:   %.loc11: %DefaultFn.type = converted %C.ref, %DefaultFn.facet [concrete = constants.%DefaultFn.facet]
// CHECK:STDOUT:   %CallFn.specific_fn: <specific function> = specific_function %CallFn.ref, @CallFn(constants.%DefaultFn.facet) [concrete = constants.%CallFn.specific_fn]
// CHECK:STDOUT:   %CallFn.call: init %empty_tuple.type = call %CallFn.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %DefaultMethod.type: type = facet_type <@DefaultMethod> [concrete]
// CHECK:STDOUT:   %.3e0: <witness> = impl_self_witness %C, @DefaultMethod [concrete]
// CHECK:STDOUT:   %DefaultMethod.impl_witness: <witness> = impl_witness @C.as.DefaultMethod.impl.%DefaultMethod.impl_witness_table [concrete]
// CHECK:STDOUT:   %DefaultMethod.facet: %DefaultMethod.type = facet_value %C, (%DefaultMethod.impl_witness) [concrete]
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.type.171: type = fn_type @DefaultMethod.WithSelf.M, @DefaultMethod.WithSelf(%DefaultMethod.facet) [concrete]
// CHECK:STDOUT:   %DefaultMethod.WithSelf.M.076: %DefaultMethod.WithSelf.M.type.171 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallMethod.specific_fn: <specific function> = specific_function %CallMethod, @CallMethod(%DefaultMethod.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc11_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.115, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.DefaultMethod: type = import_ref Main//default_interfaces, DefaultMethod, loaded [concrete = constants.%DefaultMethod.type]
// CHECK:STDOUT:   %Main.CallMethod: %CallMethod.type = import_ref Main//default_interfaces, CallMethod, loaded [concrete = constants.%CallMethod]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.DefaultMethod.impl: %Self.ref as %DefaultMethod.ref {
// CHECK:STDOUT:   %DefaultMethod.impl_witness_table = impl_witness_table (constants.%DefaultMethod.WithSelf.M.076), @C.as.DefaultMethod.impl [concrete]
// CHECK:STDOUT:   %DefaultMethod.impl_witness: <witness> = impl_witness %DefaultMethod.impl_witness_table [concrete = constants.%DefaultMethod.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   extend %DefaultMethod.ref
// CHECK:STDOUT:   witness = %DefaultMethod.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.DefaultMethod.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %DefaultMethod.ref: type = name_ref DefaultMethod, imports.%Main.DefaultMethod [concrete = constants.%DefaultMethod.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.DefaultMethod.impl.%Self.ref, @DefaultMethod [concrete = constants.%.3e0]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .DefaultMethod = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_15.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc11_15.3: init %C to %.loc11_15.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_15.4: init %C = converted %.loc11_15.1, %.loc11_15.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_15.5: ref %C = temporary %.loc11_15.2, %.loc11_15.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc11_15.6: %C = acquire_value %.loc11_15.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc11_15.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallMethod.ref: %CallMethod.type = name_ref CallMethod, imports.%Main.CallMethod [concrete = constants.%CallMethod]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %DefaultMethod.facet.loc12_15.1: %DefaultMethod.type = facet_value constants.%C, (constants.%DefaultMethod.impl_witness) [concrete = constants.%DefaultMethod.facet]
// CHECK:STDOUT:   %.loc12_15.1: %DefaultMethod.type = converted constants.%C, %DefaultMethod.facet.loc12_15.1 [concrete = constants.%DefaultMethod.facet]
// CHECK:STDOUT:   %DefaultMethod.facet.loc12_15.2: %DefaultMethod.type = facet_value constants.%C, (constants.%DefaultMethod.impl_witness) [concrete = constants.%DefaultMethod.facet]
// CHECK:STDOUT:   %.loc12_15.2: %DefaultMethod.type = converted constants.%C, %DefaultMethod.facet.loc12_15.2 [concrete = constants.%DefaultMethod.facet]
// CHECK:STDOUT:   %CallMethod.specific_fn: <specific function> = specific_function %CallMethod.ref, @CallMethod(constants.%DefaultMethod.facet) [concrete = constants.%CallMethod.specific_fn]
// CHECK:STDOUT:   %CallMethod.call: init %empty_tuple.type = call %CallMethod.specific_fn(%c.ref)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.115)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_15.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_function_overridden.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultFn.type: type = facet_type <@DefaultFn> [concrete]
// CHECK:STDOUT:   %.1d8: <witness> = impl_self_witness %C, @DefaultFn [concrete]
// CHECK:STDOUT:   %DefaultFn.impl_witness: <witness> = impl_witness @C.as.DefaultFn.impl.%DefaultFn.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.DefaultFn.impl.F.type: type = fn_type @C.as.DefaultFn.impl.F [concrete]
// CHECK:STDOUT:   %C.as.DefaultFn.impl.F: %C.as.DefaultFn.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultFn.facet: %DefaultFn.type = facet_value %C, (%DefaultFn.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFn.type: type = fn_type @CallFn [concrete]
// CHECK:STDOUT:   %CallFn: %CallFn.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFn.specific_fn: <specific function> = specific_function %CallFn, @CallFn(%DefaultFn.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.DefaultFn: type = import_ref Main//default_interfaces, DefaultFn, loaded [concrete = constants.%DefaultFn.type]
// CHECK:STDOUT:   %Main.CallFn: %CallFn.type = import_ref Main//default_interfaces, CallFn, loaded [concrete = constants.%CallFn]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.DefaultFn.impl: %Self.ref as %DefaultFn.ref {
// CHECK:STDOUT:   %C.as.DefaultFn.impl.F.decl: %C.as.DefaultFn.impl.F.type = fn_decl @C.as.DefaultFn.impl.F [concrete = constants.%C.as.DefaultFn.impl.F] {} {}
// CHECK:STDOUT:   %DefaultFn.impl_witness_table = impl_witness_table (%C.as.DefaultFn.impl.F.decl), @C.as.DefaultFn.impl [concrete]
// CHECK:STDOUT:   %DefaultFn.impl_witness: <witness> = impl_witness %DefaultFn.impl_witness_table [concrete = constants.%DefaultFn.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.DefaultFn.impl.F.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   extend %DefaultFn.ref
// CHECK:STDOUT:   witness = %DefaultFn.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.G.decl: %C.G.type = fn_decl @C.G [concrete = constants.%C.G] {} {}
// CHECK:STDOUT:   impl_decl @C.as.DefaultFn.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %DefaultFn.ref: type = name_ref DefaultFn, imports.%Main.DefaultFn [concrete = constants.%DefaultFn.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C.as.DefaultFn.impl.%Self.ref, @DefaultFn [concrete = constants.%.1d8]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .G = %C.G.decl
// CHECK:STDOUT:   .DefaultFn = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.DefaultFn.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %G.ref: %C.G.type = name_ref G, @C.%C.G.decl [concrete = constants.%C.G]
// CHECK:STDOUT:   %C.G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallFn.ref: %CallFn.type = name_ref CallFn, imports.%Main.CallFn [concrete = constants.%CallFn]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %DefaultFn.facet: %DefaultFn.type = facet_value %C.ref, (constants.%DefaultFn.impl_witness) [concrete = constants.%DefaultFn.facet]
// CHECK:STDOUT:   %.loc14: %DefaultFn.type = converted %C.ref, %DefaultFn.facet [concrete = constants.%DefaultFn.facet]
// CHECK:STDOUT:   %CallFn.specific_fn: <specific function> = specific_function %CallFn.ref, @CallFn(constants.%DefaultFn.facet) [concrete = constants.%CallFn.specific_fn]
// CHECK:STDOUT:   %CallFn.call: init %empty_tuple.type = call %CallFn.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_method_overridden.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.36a3f7.1: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultMethod.type: type = facet_type <@DefaultMethod> [concrete]
// CHECK:STDOUT:   %.3e0: <witness> = impl_self_witness %C, @DefaultMethod [concrete]
// CHECK:STDOUT:   %DefaultMethod.impl_witness: <witness> = impl_witness @C.as.DefaultMethod.impl.%DefaultMethod.impl_witness_table [concrete]
// CHECK:STDOUT:   %self.patt.36a3f7.2: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %C.as.DefaultMethod.impl.M.type: type = fn_type @C.as.DefaultMethod.impl.M [concrete]
// CHECK:STDOUT:   %C.as.DefaultMethod.impl.M: %C.as.DefaultMethod.impl.M.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultMethod.facet: %DefaultMethod.type = facet_value %C, (%DefaultMethod.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallMethod.specific_fn: <specific function> = specific_function %CallMethod, @CallMethod(%DefaultMethod.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc14_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.115, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.DefaultMethod: type = import_ref Main//default_interfaces, DefaultMethod, loaded [concrete = constants.%DefaultMethod.type]
// CHECK:STDOUT:   %Main.CallMethod: %CallMethod.type = import_ref Main//default_interfaces, CallMethod, loaded [concrete = constants.%CallMethod]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.DefaultMethod.impl: %Self.ref as %DefaultMethod.ref {
// CHECK:STDOUT:   %C.as.DefaultMethod.impl.M.decl: %C.as.DefaultMethod.impl.M.type = fn_decl @C.as.DefaultMethod.impl.M [concrete = constants.%C.as.DefaultMethod.impl.M] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%self.param_patt.303]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.36a3f7.2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultMethod.impl_witness_table = impl_witness_table (%C.as.DefaultMethod.impl.M.decl), @C.as.DefaultMethod.impl [concrete]
// CHECK:STDOUT:   %DefaultMethod.impl_witness: <witness> = impl_witness %DefaultMethod.impl_witness_table [concrete = constants.%DefaultMethod.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .M = %C.as.DefaultMethod.impl.M.decl
// CHECK:STDOUT:   extend %DefaultMethod.ref
// CHECK:STDOUT:   witness = %DefaultMethod.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.G.decl: %C.G.type = fn_decl @C.G [concrete = constants.%C.G] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%self.param_patt.303]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.36a3f7.1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.DefaultMethod.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %DefaultMethod.ref: type = name_ref DefaultMethod, imports.%Main.DefaultMethod [concrete = constants.%DefaultMethod.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C.as.DefaultMethod.impl.%Self.ref, @DefaultMethod [concrete = constants.%.3e0]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .G = %C.G.decl
// CHECK:STDOUT:   .DefaultMethod = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.DefaultMethod.impl.M(%self.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %G.ref: %C.G.type = name_ref G, @C.%C.G.decl [concrete = constants.%C.G]
// CHECK:STDOUT:   %C.G.bound: <bound method> = bound_method %self.ref, %G.ref
// CHECK:STDOUT:   %C.G.call: init %empty_tuple.type = call %C.G.bound(%self.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_15.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14_15.3: init %C to %.loc14_15.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc14_15.4: init %C = converted %.loc14_15.1, %.loc14_15.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc14_15.5: ref %C = temporary %.loc14_15.2, %.loc14_15.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc14_15.6: %C = acquire_value %.loc14_15.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc14_15.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallMethod.ref: %CallMethod.type = name_ref CallMethod, imports.%Main.CallMethod [concrete = constants.%CallMethod]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %DefaultMethod.facet.loc15_15.1: %DefaultMethod.type = facet_value constants.%C, (constants.%DefaultMethod.impl_witness) [concrete = constants.%DefaultMethod.facet]
// CHECK:STDOUT:   %.loc15_15.1: %DefaultMethod.type = converted constants.%C, %DefaultMethod.facet.loc15_15.1 [concrete = constants.%DefaultMethod.facet]
// CHECK:STDOUT:   %DefaultMethod.facet.loc15_15.2: %DefaultMethod.type = facet_value constants.%C, (constants.%DefaultMethod.impl_witness) [concrete = constants.%DefaultMethod.facet]
// CHECK:STDOUT:   %.loc15_15.2: %DefaultMethod.type = converted constants.%C, %DefaultMethod.facet.loc15_15.2 [concrete = constants.%DefaultMethod.facet]
// CHECK:STDOUT:   %CallMethod.specific_fn: <specific function> = specific_function %CallMethod.ref, @CallMethod(constants.%DefaultMethod.facet) [concrete = constants.%CallMethod.specific_fn]
// CHECK:STDOUT:   %CallMethod.call: init %empty_tuple.type = call %CallMethod.specific_fn(%c.ref)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.115)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_15.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %FinalFn.type: type = facet_type <@FinalFn> [concrete]
// CHECK:STDOUT:   %.55e: <witness> = impl_self_witness %C, @FinalFn [concrete]
// CHECK:STDOUT:   %FinalFn.impl_witness: <witness> = impl_witness @C.as.FinalFn.impl.%FinalFn.impl_witness_table [concrete]
// CHECK:STDOUT:   %FinalFn.facet: %FinalFn.type = facet_value %C, (%FinalFn.impl_witness) [concrete]
// CHECK:STDOUT:   %FinalFn.WithSelf.F.type.87d: type = fn_type @FinalFn.WithSelf.F, @FinalFn.WithSelf(%FinalFn.facet) [concrete]
// CHECK:STDOUT:   %FinalFn.WithSelf.F.3f7: %FinalFn.WithSelf.F.type.87d = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFn.type: type = fn_type @CallFn [concrete]
// CHECK:STDOUT:   %CallFn: %CallFn.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallFn.specific_fn: <specific function> = specific_function %CallFn, @CallFn(%FinalFn.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FinalFn: type = import_ref Main//final_interfaces, FinalFn, loaded [concrete = constants.%FinalFn.type]
// CHECK:STDOUT:   %Main.CallFn: %CallFn.type = import_ref Main//final_interfaces, CallFn, loaded [concrete = constants.%CallFn]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.FinalFn.impl: %Self.ref as %FinalFn.ref {
// CHECK:STDOUT:   %FinalFn.impl_witness_table = impl_witness_table (constants.%FinalFn.WithSelf.F.3f7), @C.as.FinalFn.impl [concrete]
// CHECK:STDOUT:   %FinalFn.impl_witness: <witness> = impl_witness %FinalFn.impl_witness_table [concrete = constants.%FinalFn.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %FinalFn.ref
// CHECK:STDOUT:   witness = %FinalFn.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.FinalFn.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %FinalFn.ref: type = name_ref FinalFn, imports.%Main.FinalFn [concrete = constants.%FinalFn.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.FinalFn.impl.%Self.ref, @FinalFn [concrete = constants.%.55e]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .FinalFn = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CallFn.ref: %CallFn.type = name_ref CallFn, imports.%Main.CallFn [concrete = constants.%CallFn]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %FinalFn.facet: %FinalFn.type = facet_value %C.ref, (constants.%FinalFn.impl_witness) [concrete = constants.%FinalFn.facet]
// CHECK:STDOUT:   %.loc11: %FinalFn.type = converted %C.ref, %FinalFn.facet [concrete = constants.%FinalFn.facet]
// CHECK:STDOUT:   %CallFn.specific_fn: <specific function> = specific_function %CallFn.ref, @CallFn(constants.%FinalFn.facet) [concrete = constants.%CallFn.specific_fn]
// CHECK:STDOUT:   %CallFn.call: init %empty_tuple.type = call %CallFn.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %FinalMethod.type: type = facet_type <@FinalMethod> [concrete]
// CHECK:STDOUT:   %.864: <witness> = impl_self_witness %C, @FinalMethod [concrete]
// CHECK:STDOUT:   %FinalMethod.impl_witness: <witness> = impl_witness @C.as.FinalMethod.impl.%FinalMethod.impl_witness_table [concrete]
// CHECK:STDOUT:   %FinalMethod.facet: %FinalMethod.type = facet_value %C, (%FinalMethod.impl_witness) [concrete]
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.type.7a1: type = fn_type @FinalMethod.WithSelf.M, @FinalMethod.WithSelf(%FinalMethod.facet) [concrete]
// CHECK:STDOUT:   %FinalMethod.WithSelf.M.ed4: %FinalMethod.WithSelf.M.type.7a1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %CallMethod.type: type = fn_type @CallMethod [concrete]
// CHECK:STDOUT:   %CallMethod: %CallMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallMethod.specific_fn: <specific function> = specific_function %CallMethod, @CallMethod(%FinalMethod.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc11_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.115, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.FinalMethod: type = import_ref Main//final_interfaces, FinalMethod, loaded [concrete = constants.%FinalMethod.type]
// CHECK:STDOUT:   %Main.CallMethod: %CallMethod.type = import_ref Main//final_interfaces, CallMethod, loaded [concrete = constants.%CallMethod]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.FinalMethod.impl: %Self.ref as %FinalMethod.ref {
// CHECK:STDOUT:   %FinalMethod.impl_witness_table = impl_witness_table (constants.%FinalMethod.WithSelf.M.ed4), @C.as.FinalMethod.impl [concrete]
// CHECK:STDOUT:   %FinalMethod.impl_witness: <witness> = impl_witness %FinalMethod.impl_witness_table [concrete = constants.%FinalMethod.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   extend %FinalMethod.ref
// CHECK:STDOUT:   witness = %FinalMethod.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.FinalMethod.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %FinalMethod.ref: type = name_ref FinalMethod, imports.%Main.FinalMethod [concrete = constants.%FinalMethod.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.FinalMethod.impl.%Self.ref, @FinalMethod [concrete = constants.%.864]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .FinalMethod = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_15.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc11_15.3: init %C to %.loc11_15.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_15.4: init %C = converted %.loc11_15.1, %.loc11_15.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_15.5: ref %C = temporary %.loc11_15.2, %.loc11_15.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc11_15.6: %C = acquire_value %.loc11_15.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc11_15.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallMethod.ref: %CallMethod.type = name_ref CallMethod, imports.%Main.CallMethod [concrete = constants.%CallMethod]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %FinalMethod.facet.loc12_15.1: %FinalMethod.type = facet_value constants.%C, (constants.%FinalMethod.impl_witness) [concrete = constants.%FinalMethod.facet]
// CHECK:STDOUT:   %.loc12_15.1: %FinalMethod.type = converted constants.%C, %FinalMethod.facet.loc12_15.1 [concrete = constants.%FinalMethod.facet]
// CHECK:STDOUT:   %FinalMethod.facet.loc12_15.2: %FinalMethod.type = facet_value constants.%C, (constants.%FinalMethod.impl_witness) [concrete = constants.%FinalMethod.facet]
// CHECK:STDOUT:   %.loc12_15.2: %FinalMethod.type = converted constants.%C, %FinalMethod.facet.loc12_15.2 [concrete = constants.%FinalMethod.facet]
// CHECK:STDOUT:   %CallMethod.specific_fn: <specific function> = specific_function %CallMethod.ref, @CallMethod(constants.%FinalMethod.facet) [concrete = constants.%CallMethod.specific_fn]
// CHECK:STDOUT:   %CallMethod.call: init %empty_tuple.type = call %CallMethod.specific_fn(%c.ref)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.115)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_15.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
