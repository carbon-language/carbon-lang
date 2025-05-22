// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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

fn AccessDerived[T:! type](x: Derived(T)) -> T {
  //@dump-sem-ir-begin
  return x.d;
  //@dump-sem-ir-end
}

fn AccessBase[T:! type](x: Derived(T)) -> T {
  //@dump-sem-ir-begin
  return x.b;
  //@dump-sem-ir-end
}

fn AccessConcrete(x: Derived(i32)) -> i32 {
  //@dump-sem-ir-begin
  return x.b;
  //@dump-sem-ir-end
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.9af: type = unbound_element_type %Base.370, %T [symbolic]
// CHECK:STDOUT:   %Derived.85c: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.6d2: type = unbound_element_type %Derived.85c, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Derived.115: type = class_type @Derived, @Derived(%i32) [concrete]
// CHECK:STDOUT:   %Base.10a: type = class_type @Base, @Base(%i32) [concrete]
// CHECK:STDOUT:   %Base.elem.a98: type = unbound_element_type %Base.10a, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessDerived(%T.loc13_18.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived.loc13_40.2, %T.loc13_18.2 [symbolic = %Derived.elem (constants.%Derived.elem.6d2)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %T.loc13_18.2 [symbolic = %require_complete.loc15 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessDerived.%Derived.loc13_40.2 (%Derived.85c)) -> @AccessDerived.%T.loc13_18.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessDerived.%Derived.loc13_40.2 (%Derived.85c) = name_ref x, %x
// CHECK:STDOUT:     %d.ref: @AccessDerived.%Derived.elem (%Derived.elem.6d2) = name_ref d, @Derived.%.loc10 [concrete = @Derived.%.loc10]
// CHECK:STDOUT:     %.loc15_11.1: ref @AccessDerived.%T.loc13_18.2 (%T) = class_element_access %x.ref, element1
// CHECK:STDOUT:     %.loc15_11.2: @AccessDerived.%T.loc13_18.2 (%T) = bind_value %.loc15_11.1
// CHECK:STDOUT:     return %.loc15_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessBase(%T.loc19_15.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc19_15.2) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc21_11: <witness> = require_complete_type %Base [symbolic = %require_complete.loc21_11 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc19_15.2 [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %require_complete.loc21_13: <witness> = require_complete_type %T.loc19_15.2 [symbolic = %require_complete.loc21_13 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessBase.%Derived.loc19_37.2 (%Derived.85c)) -> @AccessBase.%T.loc19_15.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessBase.%Derived.loc19_37.2 (%Derived.85c) = name_ref x, %x
// CHECK:STDOUT:     %b.ref: @AccessBase.%Base.elem (%Base.elem.9af) = name_ref b, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:     %.loc21_11.1: ref @AccessBase.%Base (%Base.370) = class_element_access %x.ref, element0
// CHECK:STDOUT:     %.loc21_11.2: ref @AccessBase.%Base (%Base.370) = converted %x.ref, %.loc21_11.1
// CHECK:STDOUT:     %.loc21_11.3: ref @AccessBase.%T.loc19_15.2 (%T) = class_element_access %.loc21_11.2, element0
// CHECK:STDOUT:     %.loc21_11.4: @AccessBase.%T.loc19_15.2 (%T) = bind_value %.loc21_11.3
// CHECK:STDOUT:     return %.loc21_11.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessConcrete(%x.param: %Derived.115) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Derived.115 = name_ref x, %x
// CHECK:STDOUT:   %b.ref: %Base.elem.a98 = name_ref b, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:   %.loc27_11.1: ref %Base.10a = class_element_access %x.ref, element0
// CHECK:STDOUT:   %.loc27_11.2: ref %Base.10a = converted %x.ref, %.loc27_11.1
// CHECK:STDOUT:   %.loc27_11.3: ref %i32 = class_element_access %.loc27_11.2, element0
// CHECK:STDOUT:   %.loc27_11.4: %i32 = bind_value %.loc27_11.3
// CHECK:STDOUT:   return %.loc27_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc13_18.2 => constants.%T
// CHECK:STDOUT:   %Derived.loc13_40.2 => constants.%Derived.85c
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.423
// CHECK:STDOUT:   %pattern_type.loc13_43 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessBase(constants.%T) {
// CHECK:STDOUT:   %T.loc19_15.2 => constants.%T
// CHECK:STDOUT:   %Derived.loc19_37.2 => constants.%Derived.85c
// CHECK:STDOUT:   %pattern_type.loc19_25 => constants.%pattern_type.423
// CHECK:STDOUT:   %pattern_type.loc19_40 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
