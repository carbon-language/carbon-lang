// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/modifiers.carbon

// --- default_method.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface Default {
  default fn Method() { }
}

class C {}

impl C as Default { }

fn F(generic d: Default) {
  d.Method();
}

fn Run() {
  F(C);
}
//@dump-sem-ir-end

// --- default_method_self.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface Default {
  default fn Method(unused self) { }
}

class C {}

impl C as Default { }

fn F[T: Default](t: T) {
  t.Method();
}

fn Run() {
  // CHECK:STDERR: default_method_self.carbon:[[@LINE+4]]:14: error: `Core.Destroy` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   let c: C = {};
  // CHECK:STDERR:              ^~
  // CHECK:STDERR:
  let c: C = {};
  F(c);
}
//@dump-sem-ir-end

// --- default_method_for_cpp_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class C {};
''';

//@dump-sem-ir-begin
interface Default {
  default fn Method() { }
}

impl Cpp.C as Default { }

fn F(generic d: Default) {
  d.Method();
}

fn Run() {
  F(Cpp.C);
}
//@dump-sem-ir-end

// --- default_method_self_for_cpp_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class C {};
''';

//@dump-sem-ir-begin
interface Default {
  default fn Method(unused self) { }
}

impl Cpp.C as Default { }

fn F[T: Default](t: T) {
  t.Method();
}

fn Run() {
  // CHECK:STDERR: default_method_self_for_cpp_type.carbon:[[@LINE+4]]:18: error: `Core.Destroy` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   let c: Cpp.C = {};
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR:
  let c: Cpp.C = {};
  F(c);
}
//@dump-sem-ir-end

// --- fail_todo_default_deferred_modifiers.carbon

interface Default {
  // CHECK:STDERR: fail_todo_default_deferred_modifiers.carbon:[[@LINE+4]]:3: error: `default` modifier requires a default implementation [ModifierFinalRequiresDefaultImpl]
  // CHECK:STDERR:   default fn M1();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn M1();
  // CHECK:STDERR: fail_todo_default_deferred_modifiers.carbon:[[@LINE+4]]:3: error: `default` modifier requires a default implementation [ModifierFinalRequiresDefaultImpl]
  // CHECK:STDERR:   default fn M2(self);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn M2(self);
}

// CHECK:STDOUT: --- default_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.9c3: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.6e3: %Default.WithSelf.Method.type.9c3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.assoc_type: type = assoc_entity_type @Default [concrete]
// CHECK:STDOUT:   %assoc0: %Default.assoc_type = assoc_entity element0, @Default.WithSelf.%Default.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.515: <witness> = impl_self_witness %C, @Default [concrete]
// CHECK:STDOUT:   %Default.impl_witness: <witness> = impl_witness @C.as.Default.impl.%Default.impl_witness_table [concrete]
// CHECK:STDOUT:   %Default.facet: %Default.type = facet_value %C, (%Default.impl_witness) [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.77c: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%Default.facet) [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.4bf: %Default.WithSelf.Method.type.77c = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Default.type [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type = symbolic_binding_pattern d, 0 [symbolic]
// CHECK:STDOUT:   %d: %Default.type = symbolic_binding d, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.b3f: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%d) [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.e25: %Default.WithSelf.Method.type.b3f = struct_value () [symbolic]
// CHECK:STDOUT:   %d.as_type: type = facet_access_type %d [symbolic]
// CHECK:STDOUT:   %Default.lookup_impl_witness: <witness> = lookup_impl_witness %d, @Default [symbolic]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Default.WithSelf.Method.type.b3f, %d [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.0ea = impl_witness_access %Default.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Default.WithSelf.Method(%d) [symbolic]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Default.facet) [concrete]
// CHECK:STDOUT:   %.d12: type = fn_type_with_self_type %Default.WithSelf.Method.type.77c, %Default.facet [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.specific_fn: <specific function> = specific_function %Default.WithSelf.Method.6e3, @Default.WithSelf.Method(%Self) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Default.decl: type = interface_decl @Default [concrete = constants.%Default.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Default.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Default.ref: type = name_ref Default, file.%Default.decl [concrete = constants.%Default.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @C.as.Default.impl.%C.ref, @Default [concrete = constants.%.515]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %d.patt.loc13_15.1: %pattern_type = symbolic_binding_pattern d, 0 [symbolic = %d.patt.loc13_15.2 (constants.%d.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %Default.ref [concrete = constants.%Default.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Default.ref: type = name_ref Default, file.%Default.decl [concrete = constants.%Default.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d.loc13_15.2: %Default.type = symbolic_binding d, 0 [symbolic = %d.loc13_15.1 (constants.%d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Default {
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Default.WithSelf.decl = interface_with_self_decl @Default [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Default.WithSelf.Method.decl: @Default.WithSelf.%Default.WithSelf.Method.type (%Default.WithSelf.Method.type.9c3) = fn_decl @Default.WithSelf.Method [symbolic = @Default.WithSelf.%Default.WithSelf.Method (constants.%Default.WithSelf.Method.6e3)] {} {}
// CHECK:STDOUT:   %assoc0: %Default.assoc_type = assoc_entity element0, %Default.WithSelf.Method.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Method = @Default.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Default.WithSelf.%Default.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Default.impl: %C.ref as %Default.ref {
// CHECK:STDOUT:   %Default.impl_witness_table = impl_witness_table (@Default.WithSelf.%Default.WithSelf.Method.decl), @C.as.Default.impl [concrete]
// CHECK:STDOUT:   %Default.impl_witness: <witness> = impl_witness %Default.impl_witness_table [concrete = constants.%Default.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Method = <poisoned>
// CHECK:STDOUT:   extend %Default.ref
// CHECK:STDOUT:   witness = %Default.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Default.WithSelf.Method(@Default.%Self: %Default.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%d.loc13_15.2: %Default.type) {
// CHECK:STDOUT:   %d.patt.loc13_15.2: %pattern_type = symbolic_binding_pattern d, 0 [symbolic = %d.patt.loc13_15.2 (constants.%d.patt)]
// CHECK:STDOUT:   %d.loc13_15.1: %Default.type = symbolic_binding d, 0 [symbolic = %d.loc13_15.1 (constants.%d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %d.as_type.loc14_4.2: type = facet_access_type %d.loc13_15.1 [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:   %Default.WithSelf.Method.type: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%d.loc13_15.1) [symbolic = %Default.WithSelf.Method.type (constants.%Default.WithSelf.Method.type.b3f)]
// CHECK:STDOUT:   %.loc14_4.2: type = fn_type_with_self_type %Default.WithSelf.Method.type, %d.loc13_15.1 [symbolic = %.loc14_4.2 (constants.%.0ea)]
// CHECK:STDOUT:   %Default.lookup_impl_witness: <witness> = lookup_impl_witness %d.loc13_15.1, @Default [symbolic = %Default.lookup_impl_witness (constants.%Default.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @F.%.loc14_4.2 (%.0ea) = impl_witness_access %Default.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @Default.WithSelf.Method(%d.loc13_15.1) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %d.ref: %Default.type = name_ref d, %d.loc13_15.2 [symbolic = %d.loc13_15.1 (constants.%d)]
// CHECK:STDOUT:     %d.as_type.loc14_4.1: type = facet_access_type %d.ref [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:     %.loc14_4.1: type = converted %d.ref, %d.as_type.loc14_4.1 [symbolic = %d.as_type.loc14_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:     %Method.ref: %Default.assoc_type = name_ref Method, @Default.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @F.%.loc14_4.2 (%.0ea) = impl_witness_access constants.%Default.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @Default.WithSelf.Method(constants.%d) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Default.WithSelf.Method.call: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Default.facet: %Default.type = facet_value %C.ref, (constants.%Default.impl_witness) [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %.loc18: %Default.type = converted %C.ref, %Default.facet [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Default.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.9c3
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.6e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%Default.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Default.facet
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.77c
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.4bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%Default.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%d) {
// CHECK:STDOUT:   %d.patt.loc13_15.2 => constants.%d.patt
// CHECK:STDOUT:   %d.loc13_15.1 => constants.%d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%d
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.b3f
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.e25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Default.facet) {
// CHECK:STDOUT:   %d.patt.loc13_15.2 => constants.%d.patt
// CHECK:STDOUT:   %d.loc13_15.1 => constants.%Default.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %d.as_type.loc14_4.2 => constants.%C
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.77c
// CHECK:STDOUT:   %.loc14_4.2 => constants.%.d12
// CHECK:STDOUT:   %Default.lookup_impl_witness => constants.%Default.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc14_4.2 => constants.%Default.WithSelf.Method.6e3
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2 => constants.%Default.WithSelf.Method.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_method_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.884: %pattern_type.a63 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.305: %pattern_type.a63 = at_binding_pattern self, %self.param_patt.884 [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.9c3: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.6e3: %Default.WithSelf.Method.type.9c3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.assoc_type: type = assoc_entity_type @Default [concrete]
// CHECK:STDOUT:   %assoc0: %Default.assoc_type = assoc_entity element0, @Default.WithSelf.%Default.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %require_complete.5d5: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.515: <witness> = impl_self_witness %C, @Default [concrete]
// CHECK:STDOUT:   %Default.impl_witness: <witness> = impl_witness @C.as.Default.impl.%Default.impl_witness_table [concrete]
// CHECK:STDOUT:   %Default.facet: %Default.type = facet_value %C, (%Default.impl_witness) [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.77c: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%Default.facet) [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.4bf: %Default.WithSelf.Method.type.77c = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.49a: %pattern_type.98b = at_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.af7: type = pattern_type %Default.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.af7 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.0d8: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt.805: %pattern_type.0d8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.20d: %pattern_type.0d8 = at_binding_pattern t, %t.param_patt.805 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.44a: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.b3f: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.e25: %Default.WithSelf.Method.type.b3f = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Default [symbolic]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Default.WithSelf.Method.type.b3f, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.0ea = impl_witness_access %Default.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.29f: %pattern_type.0d8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2d: %pattern_type.0d8 = at_binding_pattern self, %self.param_patt.29f [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Default.WithSelf.Method(%T) [symbolic]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %t.param_patt.5f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.c3d: %pattern_type.98b = at_binding_pattern t, %t.param_patt.5f9 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Default.facet) [concrete]
// CHECK:STDOUT:   %.d12: type = fn_type_with_self_type %Default.WithSelf.Method.type.77c, %Default.facet [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.specific_fn: <specific function> = specific_function %Default.WithSelf.Method.6e3, @Default.WithSelf.Method(%Self) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Default.decl: type = interface_decl @Default [concrete = constants.%Default.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Default.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Default.ref: type = name_ref Default, file.%Default.decl [concrete = constants.%Default.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @C.as.Default.impl.%C.ref, @Default [concrete = constants.%.515]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc13_7.1: %pattern_type.af7 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc13_19.1: @F.%pattern_type (%pattern_type.0d8) = value_param_pattern [symbolic = %t.param_patt.loc13_19.2 (constants.%t.param_patt.805)]
// CHECK:STDOUT:     %t.patt.loc13_19.1: @F.%pattern_type (%pattern_type.0d8) = at_binding_pattern t, %t.param_patt.loc13_19.1 [symbolic = %t.patt.loc13_19.2 (constants.%t.patt.20d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_9: type = splice_block %Default.ref [concrete = constants.%Default.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Default.ref: type = name_ref Default, file.%Default.decl [concrete = constants.%Default.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_7.2: %Default.type = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc13_21.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_21.1: type = splice_block %.loc13_21.2 [symbolic = %T.as_type.loc13_21.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Default.type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc13_21.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_21.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_21.2: type = converted %T.ref, %T.as_type.loc13_21.2 [symbolic = %T.as_type.loc13_21.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc13_21.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Default {
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Default.WithSelf.decl = interface_with_self_decl @Default [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Default.WithSelf.Method.decl: @Default.WithSelf.%Default.WithSelf.Method.type (%Default.WithSelf.Method.type.9c3) = fn_decl @Default.WithSelf.Method [symbolic = @Default.WithSelf.%Default.WithSelf.Method (constants.%Default.WithSelf.Method.6e3)] {
// CHECK:STDOUT:     %self.param_patt.loc6_28.1: @Default.WithSelf.Method.%pattern_type (%pattern_type.a63) = value_param_pattern [symbolic = %self.param_patt.loc6_28.2 (constants.%self.param_patt.884)]
// CHECK:STDOUT:     %self.patt.loc6_28.1: @Default.WithSelf.Method.%pattern_type (%pattern_type.a63) = at_binding_pattern self, %self.param_patt.loc6_28.1 [symbolic = %self.patt.loc6_28.2 (constants.%self.patt.305)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Default.WithSelf.Method.%Self.as_type.loc6_28.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.2 [symbolic = %Self.as_type.loc6_28.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Default.type = name_ref Self, @Default.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc6_28.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_28.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_28.2: type = converted %Self.ref, %Self.as_type.loc6_28.2 [symbolic = %Self.as_type.loc6_28.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Default.WithSelf.Method.%Self.as_type.loc6_28.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Default.assoc_type = assoc_entity element0, %Default.WithSelf.Method.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Method = @Default.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Default.WithSelf.%Default.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Default.impl: %C.ref as %Default.ref {
// CHECK:STDOUT:   %Default.impl_witness_table = impl_witness_table (@Default.WithSelf.%Default.WithSelf.Method.decl), @C.as.Default.impl [concrete]
// CHECK:STDOUT:   %Default.impl_witness: <witness> = impl_witness %Default.impl_witness_table [concrete = constants.%Default.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Method = <poisoned>
// CHECK:STDOUT:   extend %Default.ref
// CHECK:STDOUT:   witness = %Default.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Default.WithSelf.Method(@Default.%Self: %Default.type) {
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc6_28.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_28.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc6_28.1 [symbolic = %pattern_type (constants.%pattern_type.a63)]
// CHECK:STDOUT:   %self.param_patt.loc6_28.2: @Default.WithSelf.Method.%pattern_type (%pattern_type.a63) = value_param_pattern [symbolic = %self.param_patt.loc6_28.2 (constants.%self.param_patt.884)]
// CHECK:STDOUT:   %self.patt.loc6_28.2: @Default.WithSelf.Method.%pattern_type (%pattern_type.a63) = at_binding_pattern self, %self.param_patt.loc6_28.2 [symbolic = %self.patt.loc6_28.2 (constants.%self.patt.305)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc6_28.1 [symbolic = %require_complete (constants.%require_complete.5d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Default.WithSelf.Method.%Self.as_type.loc6_28.1 (%Self.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_7.2: %Default.type) {
// CHECK:STDOUT:   %T.patt.loc13_7.2: %pattern_type.af7 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc13_7.1: %Default.type = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc13_21.1: type = facet_access_type %T.loc13_7.1 [symbolic = %T.as_type.loc13_21.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc13_21.1 [symbolic = %pattern_type (constants.%pattern_type.0d8)]
// CHECK:STDOUT:   %t.param_patt.loc13_19.2: @F.%pattern_type (%pattern_type.0d8) = value_param_pattern [symbolic = %t.param_patt.loc13_19.2 (constants.%t.param_patt.805)]
// CHECK:STDOUT:   %t.patt.loc13_19.2: @F.%pattern_type (%pattern_type.0d8) = at_binding_pattern t, %t.param_patt.loc13_19.2 [symbolic = %t.patt.loc13_19.2 (constants.%t.patt.20d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc13_21.1 [symbolic = %require_complete (constants.%require_complete.44a)]
// CHECK:STDOUT:   %Default.WithSelf.Method.type: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%T.loc13_7.1) [symbolic = %Default.WithSelf.Method.type (constants.%Default.WithSelf.Method.type.b3f)]
// CHECK:STDOUT:   %.loc14_4: type = fn_type_with_self_type %Default.WithSelf.Method.type, %T.loc13_7.1 [symbolic = %.loc14_4 (constants.%.0ea)]
// CHECK:STDOUT:   %Default.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_7.1, @Default [symbolic = %Default.lookup_impl_witness (constants.%Default.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @F.%.loc14_4 (%.0ea) = impl_witness_access %Default.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @Default.WithSelf.Method(%T.loc13_7.1) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc13_21.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc13_21.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %Method.ref: %Default.assoc_type = name_ref Method, @Default.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @F.%.loc14_4 (%.0ea) = impl_witness_access constants.%Default.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc14_4: <bound method> = bound_method %t.ref, %impl.elem0.loc14_4.1
// CHECK:STDOUT:     %.loc14_12.1: %Default.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc14_12.2: %Default.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @Default.WithSelf.Method(constants.%T) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc14_12: <bound method> = bound_method %t.ref, %specific_impl_fn.loc14_4.1
// CHECK:STDOUT:     %Default.WithSelf.Method.call: init %empty_tuple.type = call %bound_method.loc14_12(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_15.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc22_15.3: init %C to %.loc22_15.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_15.4: init %C = converted %.loc22_15.1, %.loc22_15.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_15.5: ref %C = temporary %.loc22_15.2, %.loc22_15.4 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc22_15.6: %C = acquire_value %.loc22_15.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc22_15.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Default.facet.loc23_6.1: %Default.type = facet_value constants.%C, (constants.%Default.impl_witness) [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %.loc23_6.1: %Default.type = converted constants.%C, %Default.facet.loc23_6.1 [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %Default.facet.loc23_6.2: %Default.type = facet_value constants.%C, (constants.%Default.impl_witness) [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %.loc23_6.2: %Default.type = converted constants.%C, %Default.facet.loc23_6.2 [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Default.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.9c3
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.6e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc6_28.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a63
// CHECK:STDOUT:   %self.param_patt.loc6_28.2 => constants.%self.param_patt.884
// CHECK:STDOUT:   %self.patt.loc6_28.2 => constants.%self.patt.305
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%Default.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Default.facet
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.77c
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.4bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%Default.facet) {
// CHECK:STDOUT:   %Self => constants.%Default.facet
// CHECK:STDOUT:   %Self.as_type.loc6_28.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %self.param_patt.loc6_28.2 => constants.%self.param_patt.303
// CHECK:STDOUT:   %self.patt.loc6_28.2 => constants.%self.patt.49a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc13_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc13_21.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0d8
// CHECK:STDOUT:   %t.param_patt.loc13_19.2 => constants.%t.param_patt.805
// CHECK:STDOUT:   %t.patt.loc13_19.2 => constants.%t.patt.20d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.b3f
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.e25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type.loc6_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0d8
// CHECK:STDOUT:   %self.param_patt.loc6_28.2 => constants.%self.param_patt.29f
// CHECK:STDOUT:   %self.patt.loc6_28.2 => constants.%self.patt.d2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Default.facet) {
// CHECK:STDOUT:   %T.patt.loc13_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%Default.facet
// CHECK:STDOUT:   %T.as_type.loc13_21.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %t.param_patt.loc13_19.2 => constants.%t.param_patt.5f9
// CHECK:STDOUT:   %t.patt.loc13_19.2 => constants.%t.patt.c3d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.77c
// CHECK:STDOUT:   %.loc14_4 => constants.%.d12
// CHECK:STDOUT:   %Default.lookup_impl_witness => constants.%Default.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc14_4.2 => constants.%Default.WithSelf.Method.6e3
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2 => constants.%Default.WithSelf.Method.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_method_for_cpp_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.9c3: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.6e3: %Default.WithSelf.Method.type.9c3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.assoc_type: type = assoc_entity_type @Default [concrete]
// CHECK:STDOUT:   %assoc0: %Default.assoc_type = assoc_entity element0, @Default.WithSelf.%Default.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.a2f: <witness> = impl_self_witness %C, @Default [concrete]
// CHECK:STDOUT:   %Default.impl_witness: <witness> = impl_witness @C.as.Default.impl.%Default.impl_witness_table [concrete]
// CHECK:STDOUT:   %Default.facet: %Default.type = facet_value %C, (%Default.impl_witness) [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.294: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%Default.facet) [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.04d: %Default.WithSelf.Method.type.294 = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Default.type [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type = symbolic_binding_pattern d, 0 [symbolic]
// CHECK:STDOUT:   %d: %Default.type = symbolic_binding d, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.b3f: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%d) [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.e25: %Default.WithSelf.Method.type.b3f = struct_value () [symbolic]
// CHECK:STDOUT:   %d.as_type: type = facet_access_type %d [symbolic]
// CHECK:STDOUT:   %Default.lookup_impl_witness: <witness> = lookup_impl_witness %d, @Default [symbolic]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Default.WithSelf.Method.type.b3f, %d [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.0ea = impl_witness_access %Default.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Default.WithSelf.Method(%d) [symbolic]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Default.facet) [concrete]
// CHECK:STDOUT:   %.7d2: type = fn_type_with_self_type %Default.WithSelf.Method.type.294, %Default.facet [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.specific_fn: <specific function> = specific_function %Default.WithSelf.Method.6e3, @Default.WithSelf.Method(%Self) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Default.decl: type = interface_decl @Default [concrete = constants.%Default.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Default.impl [concrete] {} {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Default.ref: type = name_ref Default, file.%Default.decl [concrete = constants.%Default.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.Default.impl.%C.ref, @Default [concrete = constants.%.a2f]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %d.patt.loc15_15.1: %pattern_type = symbolic_binding_pattern d, 0 [symbolic = %d.patt.loc15_15.2 (constants.%d.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15: type = splice_block %Default.ref [concrete = constants.%Default.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Default.ref: type = name_ref Default, file.%Default.decl [concrete = constants.%Default.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d.loc15_15.2: %Default.type = symbolic_binding d, 0 [symbolic = %d.loc15_15.1 (constants.%d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Default {
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Default.WithSelf.decl = interface_with_self_decl @Default [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Default.WithSelf.Method.decl: @Default.WithSelf.%Default.WithSelf.Method.type (%Default.WithSelf.Method.type.9c3) = fn_decl @Default.WithSelf.Method [symbolic = @Default.WithSelf.%Default.WithSelf.Method (constants.%Default.WithSelf.Method.6e3)] {} {}
// CHECK:STDOUT:   %assoc0: %Default.assoc_type = assoc_entity element0, %Default.WithSelf.Method.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Method = @Default.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Default.WithSelf.%Default.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Default.impl: %C.ref as %Default.ref {
// CHECK:STDOUT:   %Default.impl_witness_table = impl_witness_table (@Default.WithSelf.%Default.WithSelf.Method.decl), @C.as.Default.impl [concrete]
// CHECK:STDOUT:   %Default.impl_witness: <witness> = impl_witness %Default.impl_witness_table [concrete = constants.%Default.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Method = <poisoned>
// CHECK:STDOUT:   extend %Default.ref
// CHECK:STDOUT:   witness = %Default.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Default.WithSelf.Method(@Default.%Self: %Default.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%d.loc15_15.2: %Default.type) {
// CHECK:STDOUT:   %d.patt.loc15_15.2: %pattern_type = symbolic_binding_pattern d, 0 [symbolic = %d.patt.loc15_15.2 (constants.%d.patt)]
// CHECK:STDOUT:   %d.loc15_15.1: %Default.type = symbolic_binding d, 0 [symbolic = %d.loc15_15.1 (constants.%d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %d.as_type.loc16_4.2: type = facet_access_type %d.loc15_15.1 [symbolic = %d.as_type.loc16_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:   %Default.WithSelf.Method.type: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%d.loc15_15.1) [symbolic = %Default.WithSelf.Method.type (constants.%Default.WithSelf.Method.type.b3f)]
// CHECK:STDOUT:   %.loc16_4.2: type = fn_type_with_self_type %Default.WithSelf.Method.type, %d.loc15_15.1 [symbolic = %.loc16_4.2 (constants.%.0ea)]
// CHECK:STDOUT:   %Default.lookup_impl_witness: <witness> = lookup_impl_witness %d.loc15_15.1, @Default [symbolic = %Default.lookup_impl_witness (constants.%Default.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @F.%.loc16_4.2 (%.0ea) = impl_witness_access %Default.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @Default.WithSelf.Method(%d.loc15_15.1) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %d.ref: %Default.type = name_ref d, %d.loc15_15.2 [symbolic = %d.loc15_15.1 (constants.%d)]
// CHECK:STDOUT:     %d.as_type.loc16_4.1: type = facet_access_type %d.ref [symbolic = %d.as_type.loc16_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:     %.loc16_4.1: type = converted %d.ref, %d.as_type.loc16_4.1 [symbolic = %d.as_type.loc16_4.2 (constants.%d.as_type)]
// CHECK:STDOUT:     %Method.ref: %Default.assoc_type = name_ref Method, @Default.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @F.%.loc16_4.2 (%.0ea) = impl_witness_access constants.%Default.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_4.1: <specific function> = specific_impl_function %impl.elem0.loc16_4.1, @Default.WithSelf.Method(constants.%d) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %Default.WithSelf.Method.call: init %empty_tuple.type = call %specific_impl_fn.loc16_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Default.facet: %Default.type = facet_value %C.ref, (constants.%Default.impl_witness) [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %.loc20: %Default.type = converted %C.ref, %Default.facet [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Default.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.9c3
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.6e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%Default.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Default.facet
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.294
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.04d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%Default.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%d) {
// CHECK:STDOUT:   %d.patt.loc15_15.2 => constants.%d.patt
// CHECK:STDOUT:   %d.loc15_15.1 => constants.%d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%d
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.b3f
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.e25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Default.facet) {
// CHECK:STDOUT:   %d.patt.loc15_15.2 => constants.%d.patt
// CHECK:STDOUT:   %d.loc15_15.1 => constants.%Default.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %d.as_type.loc16_4.2 => constants.%C
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.294
// CHECK:STDOUT:   %.loc16_4.2 => constants.%.7d2
// CHECK:STDOUT:   %Default.lookup_impl_witness => constants.%Default.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc16_4.2 => constants.%Default.WithSelf.Method.6e3
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2 => constants.%Default.WithSelf.Method.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_method_self_for_cpp_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.884: %pattern_type.a63 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.305: %pattern_type.a63 = at_binding_pattern self, %self.param_patt.884 [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.9c3: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.6e3: %Default.WithSelf.Method.type.9c3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.assoc_type: type = assoc_entity_type @Default [concrete]
// CHECK:STDOUT:   %assoc0: %Default.assoc_type = assoc_entity element0, @Default.WithSelf.%Default.WithSelf.Method.decl [concrete]
// CHECK:STDOUT:   %require_complete.5d5: <witness> = require_complete_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.a2f: <witness> = impl_self_witness %C, @Default [concrete]
// CHECK:STDOUT:   %Default.impl_witness: <witness> = impl_witness @C.as.Default.impl.%Default.impl_witness_table [concrete]
// CHECK:STDOUT:   %Default.facet: %Default.type = facet_value %C, (%Default.impl_witness) [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.294: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%Default.facet) [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.04d: %Default.WithSelf.Method.type.294 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.bbc: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.53c: %pattern_type.bbc = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.080: %pattern_type.bbc = at_binding_pattern self, %self.param_patt.53c [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.af7: type = pattern_type %Default.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.af7 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.0d8: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt.805: %pattern_type.0d8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.20d: %pattern_type.0d8 = at_binding_pattern t, %t.param_patt.805 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.44a: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.type.b3f: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Default.WithSelf.Method.e25: %Default.WithSelf.Method.type.b3f = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Default [symbolic]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Default.WithSelf.Method.type.b3f, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.0ea = impl_witness_access %Default.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.29f: %pattern_type.0d8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2d: %pattern_type.0d8 = at_binding_pattern self, %self.param_patt.29f [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Default.WithSelf.Method(%T) [symbolic]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.bbc = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.5d4: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %t.param_patt.f25: %pattern_type.bbc = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.a3d: %pattern_type.bbc = at_binding_pattern t, %t.param_patt.f25 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Default.facet) [concrete]
// CHECK:STDOUT:   %.7d2: type = fn_type_with_self_type %Default.WithSelf.Method.type.294, %Default.facet [concrete]
// CHECK:STDOUT:   %Default.WithSelf.Method.specific_fn: <specific function> = specific_function %Default.WithSelf.Method.6e3, @Default.WithSelf.Method(%Self) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Default.decl: type = interface_decl @Default [concrete = constants.%Default.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Default.impl [concrete] {} {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Default.ref: type = name_ref Default, file.%Default.decl [concrete = constants.%Default.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.Default.impl.%C.ref, @Default [concrete = constants.%.a2f]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc15_7.1: %pattern_type.af7 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc15_19.1: @F.%pattern_type (%pattern_type.0d8) = value_param_pattern [symbolic = %t.param_patt.loc15_19.2 (constants.%t.param_patt.805)]
// CHECK:STDOUT:     %t.patt.loc15_19.1: @F.%pattern_type (%pattern_type.0d8) = at_binding_pattern t, %t.param_patt.loc15_19.1 [symbolic = %t.patt.loc15_19.2 (constants.%t.patt.20d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_9: type = splice_block %Default.ref [concrete = constants.%Default.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Default.ref: type = name_ref Default, file.%Default.decl [concrete = constants.%Default.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_7.2: %Default.type = symbolic_binding T, 0 [symbolic = %T.loc15_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc15_21.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc15_21.1: type = splice_block %.loc15_21.2 [symbolic = %T.as_type.loc15_21.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Default.type = name_ref T, %T.loc15_7.2 [symbolic = %T.loc15_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc15_21.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc15_21.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_21.2: type = converted %T.ref, %T.as_type.loc15_21.2 [symbolic = %T.as_type.loc15_21.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc15_21.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Default {
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Default.WithSelf.decl = interface_with_self_decl @Default [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Default.WithSelf.Method.decl: @Default.WithSelf.%Default.WithSelf.Method.type (%Default.WithSelf.Method.type.9c3) = fn_decl @Default.WithSelf.Method [symbolic = @Default.WithSelf.%Default.WithSelf.Method (constants.%Default.WithSelf.Method.6e3)] {
// CHECK:STDOUT:     %self.param_patt.loc10_28.1: @Default.WithSelf.Method.%pattern_type (%pattern_type.a63) = value_param_pattern [symbolic = %self.param_patt.loc10_28.2 (constants.%self.param_patt.884)]
// CHECK:STDOUT:     %self.patt.loc10_28.1: @Default.WithSelf.Method.%pattern_type (%pattern_type.a63) = at_binding_pattern self, %self.param_patt.loc10_28.1 [symbolic = %self.patt.loc10_28.2 (constants.%self.patt.305)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Default.WithSelf.Method.%Self.as_type.loc10_28.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc10_28.1: type = splice_block %.loc10_28.2 [symbolic = %Self.as_type.loc10_28.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Default.type = name_ref Self, @Default.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc10_28.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_28.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc10_28.2: type = converted %Self.ref, %Self.as_type.loc10_28.2 [symbolic = %Self.as_type.loc10_28.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Default.WithSelf.Method.%Self.as_type.loc10_28.1 (%Self.as_type) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Default.assoc_type = assoc_entity element0, %Default.WithSelf.Method.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Method = @Default.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Default.WithSelf.%Default.WithSelf.Method.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Default.impl: %C.ref as %Default.ref {
// CHECK:STDOUT:   %Default.impl_witness_table = impl_witness_table (@Default.WithSelf.%Default.WithSelf.Method.decl), @C.as.Default.impl [concrete]
// CHECK:STDOUT:   %Default.impl_witness: <witness> = impl_witness %Default.impl_witness_table [concrete = constants.%Default.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Method = <poisoned>
// CHECK:STDOUT:   extend %Default.ref
// CHECK:STDOUT:   witness = %Default.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Default.WithSelf.Method(@Default.%Self: %Default.type) {
// CHECK:STDOUT:   %Self: %Default.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc10_28.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_28.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc10_28.1 [symbolic = %pattern_type (constants.%pattern_type.a63)]
// CHECK:STDOUT:   %self.param_patt.loc10_28.2: @Default.WithSelf.Method.%pattern_type (%pattern_type.a63) = value_param_pattern [symbolic = %self.param_patt.loc10_28.2 (constants.%self.param_patt.884)]
// CHECK:STDOUT:   %self.patt.loc10_28.2: @Default.WithSelf.Method.%pattern_type (%pattern_type.a63) = at_binding_pattern self, %self.param_patt.loc10_28.2 [symbolic = %self.patt.loc10_28.2 (constants.%self.patt.305)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Self.as_type.loc10_28.1 [symbolic = %require_complete (constants.%require_complete.5d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Default.WithSelf.Method.%Self.as_type.loc10_28.1 (%Self.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc15_7.2: %Default.type) {
// CHECK:STDOUT:   %T.patt.loc15_7.2: %pattern_type.af7 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc15_7.1: %Default.type = symbolic_binding T, 0 [symbolic = %T.loc15_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc15_21.1: type = facet_access_type %T.loc15_7.1 [symbolic = %T.as_type.loc15_21.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc15_21.1 [symbolic = %pattern_type (constants.%pattern_type.0d8)]
// CHECK:STDOUT:   %t.param_patt.loc15_19.2: @F.%pattern_type (%pattern_type.0d8) = value_param_pattern [symbolic = %t.param_patt.loc15_19.2 (constants.%t.param_patt.805)]
// CHECK:STDOUT:   %t.patt.loc15_19.2: @F.%pattern_type (%pattern_type.0d8) = at_binding_pattern t, %t.param_patt.loc15_19.2 [symbolic = %t.patt.loc15_19.2 (constants.%t.patt.20d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc15_21.1 [symbolic = %require_complete (constants.%require_complete.44a)]
// CHECK:STDOUT:   %Default.WithSelf.Method.type: type = fn_type @Default.WithSelf.Method, @Default.WithSelf(%T.loc15_7.1) [symbolic = %Default.WithSelf.Method.type (constants.%Default.WithSelf.Method.type.b3f)]
// CHECK:STDOUT:   %.loc16_4: type = fn_type_with_self_type %Default.WithSelf.Method.type, %T.loc15_7.1 [symbolic = %.loc16_4 (constants.%.0ea)]
// CHECK:STDOUT:   %Default.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_7.1, @Default [symbolic = %Default.lookup_impl_witness (constants.%Default.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @F.%.loc16_4 (%.0ea) = impl_witness_access %Default.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @Default.WithSelf.Method(%T.loc15_7.1) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc15_21.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc15_21.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %Method.ref: %Default.assoc_type = name_ref Method, @Default.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @F.%.loc16_4 (%.0ea) = impl_witness_access constants.%Default.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc16_4: <bound method> = bound_method %t.ref, %impl.elem0.loc16_4.1
// CHECK:STDOUT:     %.loc16_12.1: %Default.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc15_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc16_12.2: %Default.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc15_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_4.1: <specific function> = specific_impl_function %impl.elem0.loc16_4.1, @Default.WithSelf.Method(constants.%T) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc16_12: <bound method> = bound_method %t.ref, %specific_impl_fn.loc16_4.1
// CHECK:STDOUT:     %Default.WithSelf.Method.call: init %empty_tuple.type = call %bound_method.loc16_12(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_19.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc24_19.3: init %C to %.loc24_19.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_19.4: init %C = converted %.loc24_19.1, %.loc24_19.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_19.5: ref %C = temporary %.loc24_19.2, %.loc24_19.4 [concrete = constants.%.5d4]
// CHECK:STDOUT:   %.loc24_19.6: %C = acquire_value %.loc24_19.5 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc24_13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc24_19.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.bbc = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Default.facet.loc25_6.1: %Default.type = facet_value constants.%C, (constants.%Default.impl_witness) [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %.loc25_6.1: %Default.type = converted constants.%C, %Default.facet.loc25_6.1 [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %Default.facet.loc25_6.2: %Default.type = facet_value constants.%C, (constants.%Default.impl_witness) [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %.loc25_6.2: %Default.type = converted constants.%C, %Default.facet.loc25_6.2 [concrete = constants.%Default.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Default.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.9c3
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.6e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc10_28.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a63
// CHECK:STDOUT:   %self.param_patt.loc10_28.2 => constants.%self.param_patt.884
// CHECK:STDOUT:   %self.patt.loc10_28.2 => constants.%self.patt.305
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%Default.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Default.facet
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.294
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.04d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%Default.facet) {
// CHECK:STDOUT:   %Self => constants.%Default.facet
// CHECK:STDOUT:   %Self.as_type.loc10_28.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bbc
// CHECK:STDOUT:   %self.param_patt.loc10_28.2 => constants.%self.param_patt.53c
// CHECK:STDOUT:   %self.patt.loc10_28.2 => constants.%self.patt.080
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc15_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc15_21.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0d8
// CHECK:STDOUT:   %t.param_patt.loc15_19.2 => constants.%t.param_patt.805
// CHECK:STDOUT:   %t.patt.loc15_19.2 => constants.%t.patt.20d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.b3f
// CHECK:STDOUT:   %Default.WithSelf.Method => constants.%Default.WithSelf.Method.e25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Default.WithSelf.Method(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type.loc10_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0d8
// CHECK:STDOUT:   %self.param_patt.loc10_28.2 => constants.%self.param_patt.29f
// CHECK:STDOUT:   %self.patt.loc10_28.2 => constants.%self.patt.d2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Default.facet) {
// CHECK:STDOUT:   %T.patt.loc15_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_7.1 => constants.%Default.facet
// CHECK:STDOUT:   %T.as_type.loc15_21.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bbc
// CHECK:STDOUT:   %t.param_patt.loc15_19.2 => constants.%t.param_patt.f25
// CHECK:STDOUT:   %t.patt.loc15_19.2 => constants.%t.patt.a3d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Default.WithSelf.Method.type => constants.%Default.WithSelf.Method.type.294
// CHECK:STDOUT:   %.loc16_4 => constants.%.7d2
// CHECK:STDOUT:   %Default.lookup_impl_witness => constants.%Default.impl_witness
// CHECK:STDOUT:   %impl.elem0.loc16_4.2 => constants.%Default.WithSelf.Method.6e3
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2 => constants.%Default.WithSelf.Method.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
