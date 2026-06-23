// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method/call_without_method_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method/call_without_method_syntax.carbon

// --- call_without_method_syntax.carbon
library "[[@TEST_NAME]]";

class Point {
  fn Get(self) -> i32 { return self.x; }
  fn First(self, unused other: Point) -> i32 { return self.x; }

  var x: i32;
}

// A method is a function whose first parameter is `self`, so it can be called
// as an ordinary function by passing the receiver as the first explicit
// argument. `Point.Get(p)` is equivalent to `p.Get()`.
fn CallGet(p: Point) -> i32 {
  return Point.Get(p);
}

// The remaining explicit arguments follow the receiver.
fn CallFirst(p: Point, q: Point) -> i32 {
  return Point.First(p, q);
}

// The same works through an alias of the method.
fn CallViaAlias(p: Point) -> i32 {
  alias A = Point.Get;
  return A(p);
}

// CHECK:STDOUT: --- call_without_method_syntax.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Point: type = class_type @Point [concrete]
// CHECK:STDOUT:   %pattern_type.9d8: type = pattern_type %Point [concrete]
// CHECK:STDOUT:   %self.param_patt.b38: %pattern_type.9d8 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.60a: %pattern_type.9d8 = at_binding_pattern self, %self.param_patt.b38 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %Point.Get.type: type = fn_type @Point.Get [concrete]
// CHECK:STDOUT:   %Point.Get: %Point.Get.type = struct_value () [concrete]
// CHECK:STDOUT:   %other.param_patt: %pattern_type.9d8 = value_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt: %pattern_type.9d8 = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:   %Point.First.type: type = fn_type @Point.First [concrete]
// CHECK:STDOUT:   %Point.First: %Point.First.type = struct_value () [concrete]
// CHECK:STDOUT:   %Point.elem: type = unbound_element_type %Point, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.0c6: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %p.param_patt: %pattern_type.9d8 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.9d8 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   %CallGet.type: type = fn_type @CallGet [concrete]
// CHECK:STDOUT:   %CallGet: %CallGet.type = struct_value () [concrete]
// CHECK:STDOUT:   %q.param_patt: %pattern_type.9d8 = value_param_pattern [concrete]
// CHECK:STDOUT:   %q.patt: %pattern_type.9d8 = at_binding_pattern q, %q.param_patt [concrete]
// CHECK:STDOUT:   %CallFirst.type: type = fn_type @CallFirst [concrete]
// CHECK:STDOUT:   %CallFirst: %CallFirst.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallViaAlias.type: type = fn_type @CallViaAlias [concrete]
// CHECK:STDOUT:   %CallViaAlias: %CallViaAlias.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Point = %Point.decl
// CHECK:STDOUT:     .CallGet = %CallGet.decl
// CHECK:STDOUT:     .CallFirst = %CallFirst.decl
// CHECK:STDOUT:     .CallViaAlias = %CallViaAlias.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Point.decl: type = class_decl @Point [concrete = constants.%Point] {} {}
// CHECK:STDOUT:   %CallGet.decl: %CallGet.type = fn_decl @CallGet [concrete = constants.%CallGet] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.9d8 = value_param_pattern [concrete = constants.%p.param_patt]
// CHECK:STDOUT:     %p.patt: %pattern_type.9d8 = at_binding_pattern p, %p.param_patt [concrete = constants.%p.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %p.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Point.ref.loc13: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:     %p: %Point = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFirst.decl: %CallFirst.type = fn_decl @CallFirst [concrete = constants.%CallFirst] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.9d8 = value_param_pattern [concrete = constants.%p.param_patt]
// CHECK:STDOUT:     %p.patt: %pattern_type.9d8 = at_binding_pattern p, %p.param_patt [concrete = constants.%p.patt]
// CHECK:STDOUT:     %q.param_patt: %pattern_type.9d8 = value_param_pattern [concrete = constants.%q.param_patt]
// CHECK:STDOUT:     %q.patt: %pattern_type.9d8 = at_binding_pattern q, %q.param_patt [concrete = constants.%q.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %p.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Point.ref.loc18_17: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:     %p: %Point = wrapper_binding p, %p.param
// CHECK:STDOUT:     %q.param: %Point = value_param call_param1
// CHECK:STDOUT:     %Point.ref.loc18_27: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:     %q: %Point = wrapper_binding q, %q.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallViaAlias.decl: %CallViaAlias.type = fn_decl @CallViaAlias [concrete = constants.%CallViaAlias] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.9d8 = value_param_pattern [concrete = constants.%p.param_patt]
// CHECK:STDOUT:     %p.patt: %pattern_type.9d8 = at_binding_pattern p, %p.param_patt [concrete = constants.%p.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %p.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Point.ref.loc23: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:     %p: %Point = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   %Point.Get.decl: %Point.Get.type = fn_decl @Point.Get [concrete = constants.%Point.Get] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d8 = value_param_pattern [concrete = constants.%self.param_patt.b38]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d8 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.60a]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_19: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %self.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %self: %Point = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Point.First.decl: %Point.First.type = fn_decl @Point.First [concrete = constants.%Point.First] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d8 = value_param_pattern [concrete = constants.%self.param_patt.b38]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d8 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.60a]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.9d8 = value_param_pattern [concrete = constants.%other.param_patt]
// CHECK:STDOUT:     %other.patt: %pattern_type.9d8 = at_binding_pattern other, %other.param_patt [concrete = constants.%other.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_42: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %self.param: %Point = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %self: %Point = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: %Point = value_param call_param1
// CHECK:STDOUT:     %Point.ref: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:     %other: %Point = wrapper_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: %Point.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x [concrete = constants.%complete_type.0c6]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT:   .Get = %Point.Get.decl
// CHECK:STDOUT:   .Point = <poisoned>
// CHECK:STDOUT:   .First = %Point.First.decl
// CHECK:STDOUT:   .x = %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Point.Get(%self.param: %Point) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Point = name_ref self, %self
// CHECK:STDOUT:   %x.ref: %Point.elem = name_ref x, @Point.%.loc7 [concrete = @Point.%.loc7]
// CHECK:STDOUT:   %.loc4_36.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc4_36.2: %i32 = acquire_value %.loc4_36.1
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc4_36.1: <bound method> = bound_method %.loc4_36.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_36.2: <bound method> = bound_method %.loc4_36.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc4_36.2(%.loc4_36.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Point.First(%self.param: %Point, %other.param: %Point) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Point = name_ref self, %self
// CHECK:STDOUT:   %x.ref: %Point.elem = name_ref x, @Point.%.loc7 [concrete = @Point.%.loc7]
// CHECK:STDOUT:   %.loc5_59.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc5_59.2: %i32 = acquire_value %.loc5_59.1
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc5_59.1: <bound method> = bound_method %.loc5_59.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_59.2: <bound method> = bound_method %.loc5_59.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc5_59.2(%.loc5_59.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGet(%p.param: %Point) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Point.ref.loc14: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %Get.ref: %Point.Get.type = name_ref Get, @Point.%Point.Get.decl [concrete = constants.%Point.Get]
// CHECK:STDOUT:   %p.ref: %Point = name_ref p, %p
// CHECK:STDOUT:   %Point.Get.call: init %i32 = call %Get.ref(%p.ref)
// CHECK:STDOUT:   return %Point.Get.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFirst(%p.param: %Point, %q.param: %Point) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Point.ref.loc19: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %First.ref: %Point.First.type = name_ref First, @Point.%Point.First.decl [concrete = constants.%Point.First]
// CHECK:STDOUT:   %p.ref: %Point = name_ref p, %p
// CHECK:STDOUT:   %q.ref: %Point = name_ref q, %q
// CHECK:STDOUT:   %Point.First.call: init %i32 = call %First.ref(%p.ref, %q.ref)
// CHECK:STDOUT:   return %Point.First.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallViaAlias(%p.param: %Point) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Point.ref.loc24: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %Get.ref: %Point.Get.type = name_ref Get, @Point.%Point.Get.decl [concrete = constants.%Point.Get]
// CHECK:STDOUT:   %A: %Point.Get.type = alias_binding A, %Get.ref [concrete = constants.%Point.Get]
// CHECK:STDOUT:   %A.ref: %Point.Get.type = name_ref A, %A [concrete = constants.%Point.Get]
// CHECK:STDOUT:   %p.ref: %Point = name_ref p, %p
// CHECK:STDOUT:   %Point.Get.call: init %i32 = call %A.ref(%p.ref)
// CHECK:STDOUT:   return %Point.Get.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
