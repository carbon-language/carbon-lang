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
// CHECK:STDOUT:   %T.228: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.228 [symbolic]
// CHECK:STDOUT:   %Derived.746: type = class_type @Derived, @Derived(%T.as_type) [symbolic]
// CHECK:STDOUT:   %pattern_type.995: type = pattern_type %Derived.746 [symbolic]
// CHECK:STDOUT:   %.9ff809.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.4b69d0.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Base.5ea: type = class_type @Base, @Base(%T.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.fc7: <witness> = require_complete_type %Base.5ea [symbolic]
// CHECK:STDOUT:   %require_complete.c28: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Derived.elem.f71: type = unbound_element_type %Derived.746, %T.as_type [symbolic]
// CHECK:STDOUT:   %Base.elem.d1b: type = unbound_element_type %Base.5ea, %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c49: <witness> = lookup_impl_witness %T.228, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2ec0c.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.228) [symbolic]
// CHECK:STDOUT:   %.12f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2ec0c.2, %T.228 [symbolic]
// CHECK:STDOUT:   %impl.elem0.cbd: %.12f = impl_witness_access %Copy.lookup_impl_witness.c49, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c5f: <specific function> = specific_impl_function %impl.elem0.cbd, @Copy.WithSelf.Op(%T.228) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessDerived(%T.loc13_19.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived.loc13_45.1, %T.as_type.loc13_45.1 [symbolic = %Derived.elem (constants.%Derived.elem.f71)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %T.as_type.loc13_45.1 [symbolic = %require_complete.loc15 (constants.%require_complete.c28)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc13_19.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2ec0c.2)]
// CHECK:STDOUT:   %.loc15_11.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc13_19.1 [symbolic = %.loc15_11.3 (constants.%.12f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_19.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c49)]
// CHECK:STDOUT:   %impl.elem0.loc15_11.2: @AccessDerived.%.loc15_11.3 (%.12f) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_11.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_11.2: <specific function> = specific_impl_function %impl.elem0.loc15_11.2, @Copy.WithSelf.Op(%T.loc13_19.1) [symbolic = %specific_impl_fn.loc15_11.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessDerived.%Derived.loc13_45.1 (%Derived.746)) -> out %return.param: @AccessDerived.%T.as_type.loc13_45.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessDerived.%Derived.loc13_45.1 (%Derived.746) = name_ref x, %x
// CHECK:STDOUT:     %d.ref: @AccessDerived.%Derived.elem (%Derived.elem.f71) = name_ref d, @Derived.%.loc10 [concrete = @Derived.%.loc10]
// CHECK:STDOUT:     %.loc15_11.1: ref @AccessDerived.%T.as_type.loc13_45.1 (%T.as_type) = class_element_access %x.ref, element1
// CHECK:STDOUT:     %.loc15_11.2: @AccessDerived.%T.as_type.loc13_45.1 (%T.as_type) = acquire_value %.loc15_11.1
// CHECK:STDOUT:     %impl.elem0.loc15_11.1: @AccessDerived.%.loc15_11.3 (%.12f) = impl_witness_access constants.%Copy.lookup_impl_witness.c49, element0 [symbolic = %impl.elem0.loc15_11.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:     %bound_method.loc15_11.1: <bound method> = bound_method %.loc15_11.2, %impl.elem0.loc15_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_11.1: <specific function> = specific_impl_function %impl.elem0.loc15_11.1, @Copy.WithSelf.Op(constants.%T.228) [symbolic = %specific_impl_fn.loc15_11.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:     %bound_method.loc15_11.2: <bound method> = bound_method %.loc15_11.2, %specific_impl_fn.loc15_11.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @AccessDerived.%T.as_type.loc13_45.1 (%T.as_type) to %.loc13_51.1 = call %bound_method.loc15_11.2(%.loc15_11.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessBase(%T.loc19_16.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.as_type.loc19_42.1) [symbolic = %Base (constants.%Base.5ea)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.as_type.loc19_42.1 [symbolic = %Base.elem (constants.%Base.elem.d1b)]
// CHECK:STDOUT:   %require_complete.loc21_11: <witness> = require_complete_type %Base [symbolic = %require_complete.loc21_11 (constants.%require_complete.fc7)]
// CHECK:STDOUT:   %require_complete.loc21_13: <witness> = require_complete_type %T.as_type.loc19_42.1 [symbolic = %require_complete.loc21_13 (constants.%require_complete.c28)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc19_16.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2ec0c.2)]
// CHECK:STDOUT:   %.loc21_11.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc19_16.1 [symbolic = %.loc21_11.5 (constants.%.12f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc19_16.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c49)]
// CHECK:STDOUT:   %impl.elem0.loc21_11.2: @AccessBase.%.loc21_11.5 (%.12f) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:   %specific_impl_fn.loc21_11.2: <specific function> = specific_impl_function %impl.elem0.loc21_11.2, @Copy.WithSelf.Op(%T.loc19_16.1) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessBase.%Derived.loc19_42.1 (%Derived.746)) -> out %return.param: @AccessBase.%T.as_type.loc19_42.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessBase.%Derived.loc19_42.1 (%Derived.746) = name_ref x, %x
// CHECK:STDOUT:     %b.ref: @AccessBase.%Base.elem (%Base.elem.d1b) = name_ref b, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:     %.loc21_11.1: ref @AccessBase.%Base (%Base.5ea) = class_element_access %x.ref, element0
// CHECK:STDOUT:     %.loc21_11.2: ref @AccessBase.%Base (%Base.5ea) = converted %x.ref, %.loc21_11.1
// CHECK:STDOUT:     %.loc21_11.3: ref @AccessBase.%T.as_type.loc19_42.1 (%T.as_type) = class_element_access %.loc21_11.2, element0
// CHECK:STDOUT:     %.loc21_11.4: @AccessBase.%T.as_type.loc19_42.1 (%T.as_type) = acquire_value %.loc21_11.3
// CHECK:STDOUT:     %impl.elem0.loc21_11.1: @AccessBase.%.loc21_11.5 (%.12f) = impl_witness_access constants.%Copy.lookup_impl_witness.c49, element0 [symbolic = %impl.elem0.loc21_11.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:     %bound_method.loc21_11.1: <bound method> = bound_method %.loc21_11.4, %impl.elem0.loc21_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc21_11.1: <specific function> = specific_impl_function %impl.elem0.loc21_11.1, @Copy.WithSelf.Op(constants.%T.228) [symbolic = %specific_impl_fn.loc21_11.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:     %bound_method.loc21_11.2: <bound method> = bound_method %.loc21_11.4, %specific_impl_fn.loc21_11.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @AccessBase.%T.as_type.loc19_42.1 (%T.as_type) to %.loc19_48.1 = call %bound_method.loc21_11.2(%.loc21_11.4)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessDerived(constants.%T.228) {
// CHECK:STDOUT:   %T.loc13_19.1 => constants.%T.228
// CHECK:STDOUT:   %T.as_type.loc13_45.1 => constants.%T.as_type
// CHECK:STDOUT:   %Derived.loc13_45.1 => constants.%Derived.746
// CHECK:STDOUT:   %pattern_type.loc13_34 => constants.%pattern_type.995
// CHECK:STDOUT:   %.loc13_51.2 => constants.%.9ff809.2
// CHECK:STDOUT:   %pattern_type.loc13_51 => constants.%pattern_type.4b69d0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessBase(constants.%T.228) {
// CHECK:STDOUT:   %T.loc19_16.1 => constants.%T.228
// CHECK:STDOUT:   %T.as_type.loc19_42.1 => constants.%T.as_type
// CHECK:STDOUT:   %Derived.loc19_42.1 => constants.%Derived.746
// CHECK:STDOUT:   %pattern_type.loc19_31 => constants.%pattern_type.995
// CHECK:STDOUT:   %.loc19_48.2 => constants.%.9ff809.2
// CHECK:STDOUT:   %pattern_type.loc19_48 => constants.%pattern_type.4b69d0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
