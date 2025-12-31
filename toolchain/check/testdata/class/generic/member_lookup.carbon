// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_lookup.carbon

// --- member_access.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  var b: T;
}

class Derived(T:! type) {
  extend base: Base(T);
  var d: T;
}

fn AccessDerived[T:! Core.Copy](x: Derived(T)) -> T {
  //@dump-sem-ir-begin
  return x.d;
  //@dump-sem-ir-end
}

fn AccessBase[T:! Core.Copy](x: Derived(T)) -> T {
  //@dump-sem-ir-begin
  return x.b;
  //@dump-sem-ir-end
}

fn AccessConcrete(x: Derived(i32)) -> i32 {
  return x.b;
}

// --- fail_no_member.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  var b: T;
}

class Derived(T:! type) {
  extend base: Base(T);
  var d: T;
}

fn AccessMissingBase[T:! type](x: Base(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Base(T)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return x.nonesuch;
}

fn AccessMissingDerived[T:! type](x: Derived(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Derived(T)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return x.nonesuch;
}

fn AccessMissingConcrete(x: Derived(i32)) -> i32 {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Derived(i32)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return x.nonesuch;
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %Derived.ad7: type = class_type @Derived, @Derived(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %pattern_type.d85: type = pattern_type %Derived.ad7 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Base.ab3: type = class_type @Base, @Base(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.b68: <witness> = require_complete_type %Base.ab3 [symbolic]
// CHECK:STDOUT:   %Derived.elem.d6f: type = unbound_element_type %Derived.ad7, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Base.elem.384: type = unbound_element_type %Base.ab3, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %.72e: type = fn_type_with_self_type %Copy.Op.type, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.07b: %.72e = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2c9: <specific function> = specific_impl_function %impl.elem0.07b, @Copy.Op(%T.035) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessDerived(%T.loc13_18.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived.loc13_45.1, %T.binding.as_type [symbolic = %Derived.elem (constants.%Derived.elem.d6f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_18.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %.loc15_11.3: type = fn_type_with_self_type constants.%Copy.Op.type, %T.loc13_18.1 [symbolic = %.loc15_11.3 (constants.%.72e)]
// CHECK:STDOUT:   %impl.elem0.loc15_11.2: @AccessDerived.%.loc15_11.3 (%.72e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_11.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_11.2: <specific function> = specific_impl_function %impl.elem0.loc15_11.2, @Copy.Op(%T.loc13_18.1) [symbolic = %specific_impl_fn.loc15_11.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessDerived.%Derived.loc13_45.1 (%Derived.ad7)) -> %return.param: @AccessDerived.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessDerived.%Derived.loc13_45.1 (%Derived.ad7) = name_ref x, %x
// CHECK:STDOUT:     %d.ref: @AccessDerived.%Derived.elem (%Derived.elem.d6f) = name_ref d, @Derived.%.loc10 [concrete = @Derived.%.loc10]
// CHECK:STDOUT:     %.loc15_11.1: ref @AccessDerived.%T.binding.as_type (%T.binding.as_type) = class_element_access %x.ref, element1
// CHECK:STDOUT:     %.loc15_11.2: @AccessDerived.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc15_11.1
// CHECK:STDOUT:     %impl.elem0.loc15_11.1: @AccessDerived.%.loc15_11.3 (%.72e) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc15_11.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:     %bound_method.loc15_11.1: <bound method> = bound_method %.loc15_11.2, %impl.elem0.loc15_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_11.1: <specific function> = specific_impl_function %impl.elem0.loc15_11.1, @Copy.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc15_11.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:     %bound_method.loc15_11.2: <bound method> = bound_method %.loc15_11.2, %specific_impl_fn.loc15_11.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.Op.call: init @AccessDerived.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc15_11.2(%.loc15_11.2) to %.loc13_48
// CHECK:STDOUT:     return %Copy.Op.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessBase(%T.loc19_15.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.binding.as_type) [symbolic = %Base (constants.%Base.ab3)]
// CHECK:STDOUT:   %require_complete.loc21_11: <witness> = require_complete_type %Base [symbolic = %require_complete.loc21_11 (constants.%require_complete.b68)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.binding.as_type [symbolic = %Base.elem (constants.%Base.elem.384)]
// CHECK:STDOUT:   %require_complete.loc21_13: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc21_13 (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc19_15.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %.loc21_11.5: type = fn_type_with_self_type constants.%Copy.Op.type, %T.loc19_15.1 [symbolic = %.loc21_11.5 (constants.%.72e)]
// CHECK:STDOUT:   %impl.elem0.loc21_11.2: @AccessBase.%.loc21_11.5 (%.72e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:   %specific_impl_fn.loc21_11.2: <specific function> = specific_impl_function %impl.elem0.loc21_11.2, @Copy.Op(%T.loc19_15.1) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessBase.%Derived.loc19_42.1 (%Derived.ad7)) -> %return.param: @AccessBase.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessBase.%Derived.loc19_42.1 (%Derived.ad7) = name_ref x, %x
// CHECK:STDOUT:     %b.ref: @AccessBase.%Base.elem (%Base.elem.384) = name_ref b, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:     %.loc21_11.1: ref @AccessBase.%Base (%Base.ab3) = class_element_access %x.ref, element0
// CHECK:STDOUT:     %.loc21_11.2: ref @AccessBase.%Base (%Base.ab3) = converted %x.ref, %.loc21_11.1
// CHECK:STDOUT:     %.loc21_11.3: ref @AccessBase.%T.binding.as_type (%T.binding.as_type) = class_element_access %.loc21_11.2, element0
// CHECK:STDOUT:     %.loc21_11.4: @AccessBase.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc21_11.3
// CHECK:STDOUT:     %impl.elem0.loc21_11.1: @AccessBase.%.loc21_11.5 (%.72e) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:     %bound_method.loc21_11.1: <bound method> = bound_method %.loc21_11.4, %impl.elem0.loc21_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc21_11.1: <specific function> = specific_impl_function %impl.elem0.loc21_11.1, @Copy.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:     %bound_method.loc21_11.2: <bound method> = bound_method %.loc21_11.4, %specific_impl_fn.loc21_11.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.Op.call: init @AccessBase.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc21_11.2(%.loc21_11.4) to %.loc19_45
// CHECK:STDOUT:     return %Copy.Op.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessDerived(constants.%T.035) {
// CHECK:STDOUT:   %T.loc13_18.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Derived.loc13_45.1 => constants.%Derived.ad7
// CHECK:STDOUT:   %pattern_type.loc13_33 => constants.%pattern_type.d85
// CHECK:STDOUT:   %pattern_type.loc13_48 => constants.%pattern_type.9b9f0c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessBase(constants.%T.035) {
// CHECK:STDOUT:   %T.loc19_15.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Derived.loc19_42.1 => constants.%Derived.ad7
// CHECK:STDOUT:   %pattern_type.loc19_30 => constants.%pattern_type.d85
// CHECK:STDOUT:   %pattern_type.loc19_45 => constants.%pattern_type.9b9f0c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
