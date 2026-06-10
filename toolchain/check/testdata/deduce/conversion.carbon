// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/conversion.carbon

// --- need_conversion.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

fn F(unused T:! type) {}

// Requires a conversion from (type, type) -> type.
fn G() {
  //@dump-sem-ir-begin
  F((A, B));
  //@dump-sem-ir-end
}

// --- need_user_conversion.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}
impl A as Core.ImplicitAs(type) {
  eval fn Convert(unused self) -> type { return B; }
}

fn F(unused T:! type) {}

// Requires a conversion from A -> type.
fn G() {
  // This should only call `Convert` once, during deduction.
  //@dump-sem-ir-begin
  F({} as A);
  //@dump-sem-ir-end
}

// --- no_conversion_during_deduction.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}
impl A as Core.ImplicitAs(B) {
  fn Convert(self) -> B;
}

fn F(unused T:! type, unused b: B) {}

fn G(a: A) {
  // This should only call the implicit conversion function from A to B once,
  // when building the call expression, not as part of deduction.
  //@dump-sem-ir-begin
  F((), a);
  //@dump-sem-ir-end
}


// CHECK:STDOUT: --- need_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%A, %B) [concrete]
// CHECK:STDOUT:   %tuple.type.1e7: type = tuple_type (%A, %B) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%tuple.type.1e7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc12_10: %tuple.type.24b = tuple_literal (%A.ref, %B.ref) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc12_11: type = converted %.loc12_10, constants.%tuple.type.1e7 [concrete = constants.%tuple.type.1e7]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%tuple.type.1e7) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- need_user_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5f1: type = facet_type <@ImplicitAs, @ImplicitAs(type)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @A.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.type.572c06.1: type = fn_type @A.as.ImplicitAs.impl.Convert.loc7_40.1 [concrete]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.dbe75c.1: %A.as.ImplicitAs.impl.Convert.type.572c06.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.5f1 = facet_value %A, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.51c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(type, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.type.572c06.2: type = fn_type @A.as.ImplicitAs.impl.Convert.loc7_40.2 [concrete]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.dbe75c.2: %A.as.ImplicitAs.impl.Convert.type.572c06.2 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %.dd5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.51c, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.bound.5ecf75.1: <bound method> = bound_method %A.val, %A.as.ImplicitAs.impl.Convert.dbe75c.2 [concrete]
// CHECK:STDOUT:   %.367: ref %A = temporary invalid, %A.val [concrete]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.bound.5ecf75.2: <bound method> = bound_method %A.val, %A.as.ImplicitAs.impl.Convert.dbe75c.1 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc16_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.367, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc16_6.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc16_6.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc16_6.3: init %A to %.loc16_6.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc16_8.1: init %A = converted %.loc16_6.1, %.loc16_6.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %impl.elem0: %.dd5 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%A.as.ImplicitAs.impl.Convert.dbe75c.2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc16_8.1, %impl.elem0 [concrete = constants.%A.as.ImplicitAs.impl.Convert.bound.5ecf75.1]
// CHECK:STDOUT:   %.loc16_8.2: ref %A = temporary %.loc16_6.2, %.loc16_8.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %.loc16_8.3: %A = acquire_value %.loc16_8.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %Convert.ref: %A.as.ImplicitAs.impl.Convert.type.572c06.1 = name_ref Convert, @A.as.ImplicitAs.impl.%A.as.ImplicitAs.impl.Convert.decl.loc7_40.1 [concrete = constants.%A.as.ImplicitAs.impl.Convert.dbe75c.1]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %.loc16_8.3, %Convert.ref [concrete = constants.%A.as.ImplicitAs.impl.Convert.bound.5ecf75.2]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.call: init type = call %A.as.ImplicitAs.impl.Convert.bound(%.loc16_8.3) [concrete = constants.%B]
// CHECK:STDOUT:   %.loc16_12.1: type = value_of_initializer %A.as.ImplicitAs.impl.Convert.call [concrete = constants.%B]
// CHECK:STDOUT:   %.loc16_12.2: type = converted %.loc16_8.1, %.loc16_12.1 [concrete = constants.%B]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%B) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.367)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_8.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_8.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- no_conversion_during_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1bc: type = facet_type <@ImplicitAs, @ImplicitAs(%B)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @A.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.type: type = fn_type @A.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert: %A.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.1bc = facet_value %A, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6c1: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%B, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %.33d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6c1, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc16_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc16_6: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %.loc16_10: type = converted %.loc16_6, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%empty_tuple.type) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %impl.elem0: %.33d = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%A.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %.loc16_9.1: ref %B = temporary_storage
// CHECK:STDOUT:   %A.as.ImplicitAs.impl.Convert.call: init %B to %.loc16_9.1 = call %bound_method(%a.ref)
// CHECK:STDOUT:   %.loc16_9.2: init %B = converted %a.ref, %A.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc16_9.3: ref %B = temporary %.loc16_9.1, %.loc16_9.2
// CHECK:STDOUT:   %.loc16_9.4: %B = acquire_value %.loc16_9.3
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc16_9.4)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc16_9.3, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc16_9.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_9.2(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
