// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/for.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/for/pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/for/pattern.carbon

// --- empty_range.carbon

library "[[@TEST_NAME]]";

class EmptyRange(T: Core.Copy) {
  fn Make() -> Self { return {}; }

  impl as Core.Iterate where .CursorType = {} and .ElementType = T {
    fn NewCursor(unused self) -> {} {
      return {};
    }
    fn Next(unused self, unused ref cursor: {}) -> Core.Optional(T) {
      return Core.Optional(T).None();
    }
  }
}

class C {
  impl as Core.Copy {
    fn Op(unused self) -> Self { return {}; }
  }
}

// --- value.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

fn Body(c: C);

fn Run() {
  //@dump-sem-ir-begin
  for (c: C in EmptyRange(C).Make()) {
    Body(c);
  }
  //@dump-sem-ir-end
}

// --- var.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

fn Body(c: C*);

fn Run() {
  //@dump-sem-ir-begin
  for (var c: C in EmptyRange(C).Make()) {
    Body(&c);
  }
  //@dump-sem-ir-end
}

// --- tuple.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

fn Body(a: bool, b: bool);

fn Run() {
  //@dump-sem-ir-begin
  for ((a: bool, b: bool) in EmptyRange((bool, bool)).Make()) {
    Body(a, b);
  }
  //@dump-sem-ir-end
}

// --- tuple_class.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

fn Body(a: C, b: C);

fn Run() {
  //@dump-sem-ir-begin
  for ((a: C, b: C) in EmptyRange((C, C)).Make()) {
    Body(a, b);
  }
  //@dump-sem-ir-end
}

// --- fail_bad_pattern.carbon

library "[[@TEST_NAME]]";

import library "empty_range";

class X {}

fn Run() {
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+3]]:7-43: error: cannot implicitly convert expression of type `C` to `X` [ConversionFailure]
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+2]]:7-43: note: type `C` does not implement interface `Core.ImplicitAs(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  for (unused x: X in EmptyRange(C).Make()) {
  }
}


// CHECK:STDOUT: --- value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %c.patt.aa8: %pattern_type.98b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = parameterized_class_name_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.b43: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.35c: %EmptyRange.Make.type.b43 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %EmptyRange.c31: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.14e: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.82d: %EmptyRange.Make.type.14e = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.82d, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.afc: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.a1f: type = unbound_element_type %Optional.afc, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.fbd: type = unbound_element_type %Optional.afc, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f7a: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.c2b: %Optional.Get.type.f7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.23a: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f8f: %Optional.HasValue.type.23a = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.e76: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.706: %EmptyRange.as.Iterate.impl.Next.type.e76 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c99: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.f76: %EmptyRange.as.Iterate.impl.NewCursor.type.c99 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.562: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.1bb: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.84b: %EmptyRange.as.Iterate.impl.NewCursor.type.1bb = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.94c: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.7e7: %EmptyRange.as.Iterate.impl.Next.type.94c = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.c31, (%Iterate.impl_witness.562) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ba2: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d3a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.bff: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ba2, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.84b, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.0fb: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d3a, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.5b4: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.7e7, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.371: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.766: %Optional.HasValue.type.371 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.5d8: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.c2f: %Optional.Get.type.5d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.606: type = unbound_element_type %Optional.5b4, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.433: type = unbound_element_type %Optional.5b4, %C [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9e7: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.766, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.c2f, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_36.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_36.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %facet_value.34a: %facet_type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_36.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.1b5: %facet_type = facet_value bool, (%DestroyImpl.impl_witness.778, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_36.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.969: %SubobjectDestroy.type = facet_value %Optional.5b4, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d97: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.969) [concrete]
// CHECK:STDOUT:   %facet_value.82e: %facet_type = facet_value %Optional.5b4, (%DestroyImpl.impl_witness.d97, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %facet_value.b01: %facet_type = facet_value %empty_struct_type, (%DestroyImpl.impl_witness.011, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_35 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0c7: %SubobjectDestroy.type = facet_value %EmptyRange.c31, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.eea: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0c7) [concrete]
// CHECK:STDOUT:   %facet_value.d50: %facet_type = facet_value %EmptyRange.c31, (%DestroyImpl.impl_witness.eea, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.5e8: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.b43) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.35c)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5b1: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.23a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f8f)]
// CHECK:STDOUT:   %Core.import_ref.93e: @Optional.%Optional.Get.type (%Optional.Get.type.f7a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.c2b)]
// CHECK:STDOUT:   %Core.import_ref.5d0: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.897]
// CHECK:STDOUT:   %Core.import_ref.2f6: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.b75]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.9b4: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.c99) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.f76)]
// CHECK:STDOUT:   %Main.import_ref.405: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.e76) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.706)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.9b4, %Main.import_ref.405), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.b75: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.897: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = value_binding_pattern c [concrete = constants.%c.patt.aa8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_27: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_27, (constants.%Copy.impl_witness.518) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_28: %Copy.type = converted %C.ref.loc10_27, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.c31]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.14e = specific_constant imports.%Main.import_ref.5e8, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.82d]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.14e = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.82d]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.c31 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.c31 to %.loc10_35.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.c31 = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.bff = impl_witness_access constants.%Iterate.impl_witness.562, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.84b]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.c31 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.0fb = impl_witness_access constants.%Iterate.impl_witness.562, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.7e7]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.5b4 = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.c31 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.5b4 to %.loc10_36.1 = call %bound_method.loc10_36.4(%.loc10_35.4, %var)
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.5b4 = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.371 = specific_constant imports.%Core.import_ref.5b1, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.766]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.371 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.766]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.5b4 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.5d8 = specific_constant imports.%Core.import_ref.93e, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.c2f]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.5d8 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.c2f]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.5b4 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_36.8 = call %bound_method.loc10_36.6(%.loc10_36.9)
// CHECK:STDOUT:   %.loc10_36.10: ref %C = temporary %.loc10_36.8, %Optional.Get.call
// CHECK:STDOUT:   %.loc10_36.11: %C = acquire_value %.loc10_36.10
// CHECK:STDOUT:   %C.ref.loc10_11: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc10_36.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%c.ref)
// CHECK:STDOUT:   %facet_value.loc10_36.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_36.12: %facet_type = converted constants.%C, %facet_value.loc10_36.1 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_36.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_36.13: %facet_type = converted constants.%C, %facet_value.loc10_36.2 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_36.10)
// CHECK:STDOUT:   %facet_value.loc10_36.3: %facet_type = facet_value constants.%Optional.5b4, (constants.%DestroyImpl.impl_witness.d97, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %.loc10_36.14: %facet_type = converted constants.%Optional.5b4, %facet_value.loc10_36.3 [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %facet_value.loc10_36.4: %facet_type = facet_value constants.%Optional.5b4, (constants.%DestroyImpl.impl_witness.d97, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %.loc10_36.15: %facet_type = converted constants.%Optional.5b4, %facet_value.loc10_36.4 [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_36.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc10_36.5: %facet_type = facet_value constants.%Optional.5b4, (constants.%DestroyImpl.impl_witness.d97, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %.loc10_36.16: %facet_type = converted constants.%Optional.5b4, %facet_value.loc10_36.5 [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %facet_value.loc10_36.6: %facet_type = facet_value constants.%Optional.5b4, (constants.%DestroyImpl.impl_witness.d97, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %.loc10_36.17: %facet_type = converted constants.%Optional.5b4, %facet_value.loc10_36.6 [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_36.2)
// CHECK:STDOUT:   %facet_value.loc10_36.7: %facet_type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %.loc10_36.18: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_36.7 [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %facet_value.loc10_36.8: %facet_type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %.loc10_36.19: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_36.8 [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   %facet_value.loc10_35.1: %facet_type = facet_value constants.%EmptyRange.c31, (constants.%DestroyImpl.impl_witness.eea, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.d50]
// CHECK:STDOUT:   %.loc10_35.5: %facet_type = converted constants.%EmptyRange.c31, %facet_value.loc10_35.1 [concrete = constants.%facet_value.d50]
// CHECK:STDOUT:   %facet_value.loc10_35.2: %facet_type = facet_value constants.%EmptyRange.c31, (constants.%DestroyImpl.impl_witness.eea, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.d50]
// CHECK:STDOUT:   %.loc10_35.6: %facet_type = converted constants.%EmptyRange.c31, %facet_value.loc10_35.2 [concrete = constants.%facet_value.d50]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_35: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc10_35.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_36.1: ref %C = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_36.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_36.2: %facet_type = converted constants.%C, %facet_value.loc10_36.1 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_36.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_36.3: %facet_type = converted constants.%C, %facet_value.loc10_36.2 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_36.1)
// CHECK:STDOUT:   %.loc10_36.4: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_36.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_36.5: %facet_type = converted bool, %facet_value.loc10_36.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_36.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_36.6: %facet_type = converted bool, %facet_value.loc10_36.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_36.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.5(%self.param: ref %Optional.5b4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.433 = name_ref value, imports.%Core.import_ref.2f6 [concrete = imports.%field_decl.b75]
// CHECK:STDOUT:   %.loc10_36.1: ref %C = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_36.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_36.2: %facet_type = converted constants.%C, %facet_value.loc10_36.1 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_36.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_36.3: %facet_type = converted constants.%C, %facet_value.loc10_36.2 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_36.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.606 = name_ref has_value, imports.%Core.import_ref.5d0 [concrete = imports.%field_decl.897]
// CHECK:STDOUT:   %.loc10_36.4: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_36.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_36.5: %facet_type = converted bool, %facet_value.loc10_36.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_36.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_36.6: %facet_type = converted bool, %facet_value.loc10_36.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_36.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_35(%self.param: ref %EmptyRange.c31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt.dd0: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt.dd0 [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = parameterized_class_name_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.b43: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.35c: %EmptyRange.Make.type.b43 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %EmptyRange.c31: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.14e: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.82d: %EmptyRange.Make.type.14e = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.82d, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.afc: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.a1f: type = unbound_element_type %Optional.afc, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.fbd: type = unbound_element_type %Optional.afc, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f7a: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.c2b: %Optional.Get.type.f7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.23a: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f8f: %Optional.HasValue.type.23a = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.e76: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.706: %EmptyRange.as.Iterate.impl.Next.type.e76 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c99: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.f76: %EmptyRange.as.Iterate.impl.NewCursor.type.c99 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.562: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.1bb: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.84b: %EmptyRange.as.Iterate.impl.NewCursor.type.1bb = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.94c: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.7e7: %EmptyRange.as.Iterate.impl.Next.type.94c = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.c31, (%Iterate.impl_witness.562) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ba2: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d3a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.bff: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ba2, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.84b, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.0fb: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d3a, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.5b4: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.7e7, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.371: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.766: %Optional.HasValue.type.371 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.5d8: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.c2f: %Optional.Get.type.5d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.606: type = unbound_element_type %Optional.5b4, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.433: type = unbound_element_type %Optional.5b4, %C [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9e7: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.766, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.c2f, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_40.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_40.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_40.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %facet_value.34a: %facet_type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.1b5: %facet_type = facet_value bool, (%DestroyImpl.impl_witness.778, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_40.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.969: %SubobjectDestroy.type = facet_value %Optional.5b4, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d97: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.969) [concrete]
// CHECK:STDOUT:   %facet_value.82e: %facet_type = facet_value %Optional.5b4, (%DestroyImpl.impl_witness.d97, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %facet_value.b01: %facet_type = facet_value %empty_struct_type, (%DestroyImpl.impl_witness.011, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_39 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0c7: %SubobjectDestroy.type = facet_value %EmptyRange.c31, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.eea: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0c7) [concrete]
// CHECK:STDOUT:   %facet_value.d50: %facet_type = facet_value %EmptyRange.c31, (%DestroyImpl.impl_witness.eea, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.5e8: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.b43) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.35c)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5b1: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.23a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f8f)]
// CHECK:STDOUT:   %Core.import_ref.93e: @Optional.%Optional.Get.type (%Optional.Get.type.f7a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.c2b)]
// CHECK:STDOUT:   %Core.import_ref.5d0: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.897]
// CHECK:STDOUT:   %Core.import_ref.2f6: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.b75]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.9b4: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.c99) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.f76)]
// CHECK:STDOUT:   %Main.import_ref.405: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.e76) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.706)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.9b4, %Main.import_ref.405), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.b75: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.897: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt.dd0]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_31: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C.ref.loc10_31, (constants.%Copy.impl_witness.518) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc10_32: %Copy.type = converted %C.ref.loc10_31, %Copy.facet [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.c31]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.14e = specific_constant imports.%Main.import_ref.5e8, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.82d]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.14e = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.82d]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.c31 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.c31 to %.loc10_39.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.c31 = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.bff = impl_witness_access constants.%Iterate.impl_witness.562, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.84b]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.c31 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.0fb = impl_witness_access constants.%Iterate.impl_witness.562, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.7e7]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.5b4 = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.c31 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.5b4 to %.loc10_40.1 = call %bound_method.loc10_40.4(%.loc10_39.4, %var)
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.5b4 = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.371 = specific_constant imports.%Core.import_ref.5b1, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.766]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.371 = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.766]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.5b4 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.5d8 = specific_constant imports.%Core.import_ref.93e, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.c2f]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.5d8 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.c2f]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8.1: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.5b4 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_8.1 = call %bound_method.loc10_40.6(%.loc10_40.8)
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr: %ptr.6b6 = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr)
// CHECK:STDOUT:   %facet_value.loc10_40.1: %facet_type = facet_value constants.%Optional.5b4, (constants.%DestroyImpl.impl_witness.d97, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %.loc10_40.9: %facet_type = converted constants.%Optional.5b4, %facet_value.loc10_40.1 [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %facet_value.loc10_40.2: %facet_type = facet_value constants.%Optional.5b4, (constants.%DestroyImpl.impl_witness.d97, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %.loc10_40.10: %facet_type = converted constants.%Optional.5b4, %facet_value.loc10_40.2 [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_40.2)
// CHECK:STDOUT:   %facet_value.loc10_8.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_8.2: %facet_type = converted constants.%C, %facet_value.loc10_8.1 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_8.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_8.3: %facet_type = converted constants.%C, %facet_value.loc10_8.2 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_8.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%c.var)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc10_40.3: %facet_type = facet_value constants.%Optional.5b4, (constants.%DestroyImpl.impl_witness.d97, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %.loc10_40.11: %facet_type = converted constants.%Optional.5b4, %facet_value.loc10_40.3 [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %facet_value.loc10_40.4: %facet_type = facet_value constants.%Optional.5b4, (constants.%DestroyImpl.impl_witness.d97, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   %.loc10_40.12: %facet_type = converted constants.%Optional.5b4, %facet_value.loc10_40.4 [concrete = constants.%facet_value.82e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_40.2)
// CHECK:STDOUT:   %facet_value.loc10_8.3: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_8.4: %facet_type = converted constants.%C, %facet_value.loc10_8.3 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_8.4: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_8.5: %facet_type = converted constants.%C, %facet_value.loc10_8.4 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_8.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%c.var)
// CHECK:STDOUT:   %facet_value.loc10_40.5: %facet_type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %.loc10_40.13: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_40.5 [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %facet_value.loc10_40.6: %facet_type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %.loc10_40.14: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_40.6 [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%var)
// CHECK:STDOUT:   %facet_value.loc10_39.1: %facet_type = facet_value constants.%EmptyRange.c31, (constants.%DestroyImpl.impl_witness.eea, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.d50]
// CHECK:STDOUT:   %.loc10_39.5: %facet_type = converted constants.%EmptyRange.c31, %facet_value.loc10_39.1 [concrete = constants.%facet_value.d50]
// CHECK:STDOUT:   %facet_value.loc10_39.2: %facet_type = facet_value constants.%EmptyRange.c31, (constants.%DestroyImpl.impl_witness.eea, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.d50]
// CHECK:STDOUT:   %.loc10_39.6: %facet_type = converted constants.%EmptyRange.c31, %facet_value.loc10_39.2 [concrete = constants.%facet_value.d50]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_39: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc10_39.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_40.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_40.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_40.3(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_40.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_40.1: ref %C = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_40.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_40.2: %facet_type = converted constants.%C, %facet_value.loc10_40.1 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_40.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_40.3: %facet_type = converted constants.%C, %facet_value.loc10_40.2 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_40.1)
// CHECK:STDOUT:   %.loc10_40.4: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_40.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_40.5: %facet_type = converted bool, %facet_value.loc10_40.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_40.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_40.6: %facet_type = converted bool, %facet_value.loc10_40.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_40.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_40.5(%self.param: ref %Optional.5b4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.433 = name_ref value, imports.%Core.import_ref.2f6 [concrete = imports.%field_decl.b75]
// CHECK:STDOUT:   %.loc10_40.1: ref %C = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_40.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_40.2: %facet_type = converted constants.%C, %facet_value.loc10_40.1 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_40.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_40.3: %facet_type = converted constants.%C, %facet_value.loc10_40.2 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_40.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.606 = name_ref has_value, imports.%Core.import_ref.5d0 [concrete = imports.%field_decl.897]
// CHECK:STDOUT:   %.loc10_40.4: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_40.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_40.5: %facet_type = converted bool, %facet_value.loc10_40.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_40.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_40.6: %facet_type = converted bool, %facet_value.loc10_40.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_40.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_39(%self.param: ref %EmptyRange.c31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt.9e2: %pattern_type.831 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt.0c3: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type.784: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %pattern_type.860: type = pattern_type %tuple.type.784 [concrete]
// CHECK:STDOUT:   %.d8a: %pattern_type.860 = tuple_pattern (%a.patt.9e2, %b.patt.0c3) [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = parameterized_class_name_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.b43: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.35c: %EmptyRange.Make.type.b43 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.52b: %tuple.type.693 = tuple_value (bool, bool) [concrete]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.74a: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.9bd(%T.f84, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.4af: %tuple.type.as.Copy.impl.Op.type.74a = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet.7af: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a4e: <witness> = impl_witness imports.%Copy.impl_witness_table.e5a, @tuple.type.as.Copy.impl.9bd(%Copy.facet.7af, %Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Copy.facet.fa2: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.a4e) [concrete]
// CHECK:STDOUT:   %EmptyRange.7d6: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.408: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.94c: %EmptyRange.Make.type.408 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.94c, @EmptyRange.Make(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.afc: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.a1f: type = unbound_element_type %Optional.afc, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.fbd: type = unbound_element_type %Optional.afc, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f7a: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.c2b: %Optional.Get.type.f7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.23a: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f8f: %Optional.HasValue.type.23a = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.e76: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.706: %EmptyRange.as.Iterate.impl.Next.type.e76 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c99: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.f76: %EmptyRange.as.Iterate.impl.NewCursor.type.c99 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.aa8: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.35f: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.54f: %EmptyRange.as.Iterate.impl.NewCursor.type.35f = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.117: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.03d: %EmptyRange.as.Iterate.impl.Next.type.117 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.7d6, (%Iterate.impl_witness.aa8) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.457: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a45: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.196: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.457, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.54f, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %.d55: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a45, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.9ca: type = class_type @Optional, @Optional(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.03d, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.e99: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.0e2: %Optional.HasValue.type.e99 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.09d: type = fn_type @Optional.Get, @Optional(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Optional.Get.7e8: %Optional.Get.type.09d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.79c: type = unbound_element_type %Optional.9ca, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.6d0: type = unbound_element_type %Optional.9ca, %tuple.type.784 [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.6f9: type = struct_type {.has_value: bool, .value: %tuple.type.784} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.0e2, @Optional.HasValue(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.7e8, @Optional.Get(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_61.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_61.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.1b5: %facet_type = facet_value bool, (%DestroyImpl.impl_witness.778, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c2f: %SubobjectDestroy.type = facet_value %tuple.type.784, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bf6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c2f) [concrete]
// CHECK:STDOUT:   %facet_value.856: %facet_type = facet_value %tuple.type.784, (%DestroyImpl.impl_witness.bf6, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_61.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.88d: %SubobjectDestroy.type = facet_value %Optional.9ca, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.11e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.88d) [concrete]
// CHECK:STDOUT:   %facet_value.d86: %facet_type = facet_value %Optional.9ca, (%DestroyImpl.impl_witness.11e, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_61.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %facet_value.b01: %facet_type = facet_value %empty_struct_type, (%DestroyImpl.impl_witness.011, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_60 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.41c: %SubobjectDestroy.type = facet_value %EmptyRange.7d6, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0c1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.41c) [concrete]
// CHECK:STDOUT:   %facet_value.396: %facet_type = facet_value %EmptyRange.7d6, (%DestroyImpl.impl_witness.0c1, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.5e8: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.b43) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.35c)]
// CHECK:STDOUT:   %Core.import_ref.08a: @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.74a) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.4af)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5a = impl_witness_table (%Core.import_ref.08a), @tuple.type.as.Copy.impl.9bd [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5b1: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.23a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f8f)]
// CHECK:STDOUT:   %Core.import_ref.93e: @Optional.%Optional.Get.type (%Optional.Get.type.f7a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.c2b)]
// CHECK:STDOUT:   %Core.import_ref.5d0: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.897]
// CHECK:STDOUT:   %Core.import_ref.2f6: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.b75]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.9b4: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.c99) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.f76)]
// CHECK:STDOUT:   %Main.import_ref.405: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.e76) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.706)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.9b4, %Main.import_ref.405), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.b75: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.897: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = value_binding_pattern a [concrete = constants.%a.patt.9e2]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete = constants.%b.patt.0c3]
// CHECK:STDOUT:     %.loc10_25: %pattern_type.860 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.d8a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %.loc10_42: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc10_48: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc10_52: %tuple.type.693 = tuple_literal (%.loc10_42, %.loc10_48) [concrete = constants.%tuple.52b]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.a4e) [concrete = constants.%Copy.facet.fa2]
// CHECK:STDOUT:   %.loc10_53: %Copy.type = converted %.loc10_52, %Copy.facet [concrete = constants.%Copy.facet.fa2]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.7d6]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.408 = specific_constant imports.%Main.import_ref.5e8, @EmptyRange(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.Make.94c]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.408 = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.94c]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.7d6 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.7d6 to %.loc10_60.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.7d6 = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.196 = impl_witness_access constants.%Iterate.impl_witness.aa8, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.54f]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.7d6 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.d55 = impl_witness_access constants.%Iterate.impl_witness.aa8, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.03d]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.9ca = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.7d6 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.9ca to %.loc10_61.1 = call %bound_method.loc10_61.4(%.loc10_60.4, %var)
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.9ca = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.e99 = specific_constant imports.%Core.import_ref.5b1, @Optional(constants.%Copy.facet.fa2) [concrete = constants.%Optional.HasValue.0e2]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.e99 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.0e2]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.fa2) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.9ca = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.09d = specific_constant imports.%Core.import_ref.93e, @Optional(constants.%Copy.facet.fa2) [concrete = constants.%Optional.Get.7e8]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.09d = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.7e8]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.fa2) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.9ca = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.784 to %.loc10_61.8 = call %bound_method.loc10_61.6(%.loc10_61.9)
// CHECK:STDOUT:   %.loc10_61.10: ref %tuple.type.784 = temporary %.loc10_61.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %.loc10_61.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref bool = tuple_access %.loc10_61.10, element1
// CHECK:STDOUT:   %.loc10_61.11: bool = acquire_value %tuple.elem0
// CHECK:STDOUT:   %.loc10_12: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %a: bool = wrapper_binding a, %.loc10_61.11
// CHECK:STDOUT:   %.loc10_61.12: bool = acquire_value %tuple.elem1
// CHECK:STDOUT:   %.loc10_21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = wrapper_binding b, %.loc10_61.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %facet_value.loc10_61.1: %facet_type = facet_value constants.%tuple.type.784, (constants.%DestroyImpl.impl_witness.bf6, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %.loc10_61.13: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.1 [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %facet_value.loc10_61.2: %facet_type = facet_value constants.%tuple.type.784, (constants.%DestroyImpl.impl_witness.bf6, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %.loc10_61.14: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.2 [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_61.10)
// CHECK:STDOUT:   %facet_value.loc10_61.3: %facet_type = facet_value constants.%Optional.9ca, (constants.%DestroyImpl.impl_witness.11e, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %.loc10_61.15: %facet_type = converted constants.%Optional.9ca, %facet_value.loc10_61.3 [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %facet_value.loc10_61.4: %facet_type = facet_value constants.%Optional.9ca, (constants.%DestroyImpl.impl_witness.11e, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %.loc10_61.16: %facet_type = converted constants.%Optional.9ca, %facet_value.loc10_61.4 [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_61.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc10_61.5: %facet_type = facet_value constants.%Optional.9ca, (constants.%DestroyImpl.impl_witness.11e, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %.loc10_61.17: %facet_type = converted constants.%Optional.9ca, %facet_value.loc10_61.5 [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %facet_value.loc10_61.6: %facet_type = facet_value constants.%Optional.9ca, (constants.%DestroyImpl.impl_witness.11e, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %.loc10_61.18: %facet_type = converted constants.%Optional.9ca, %facet_value.loc10_61.6 [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_61.2)
// CHECK:STDOUT:   %facet_value.loc10_61.7: %facet_type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %.loc10_61.19: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_61.7 [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %facet_value.loc10_61.8: %facet_type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %.loc10_61.20: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_61.8 [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   %facet_value.loc10_60.1: %facet_type = facet_value constants.%EmptyRange.7d6, (constants.%DestroyImpl.impl_witness.0c1, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.396]
// CHECK:STDOUT:   %.loc10_60.5: %facet_type = converted constants.%EmptyRange.7d6, %facet_value.loc10_60.1 [concrete = constants.%facet_value.396]
// CHECK:STDOUT:   %facet_value.loc10_60.2: %facet_type = facet_value constants.%EmptyRange.7d6, (constants.%DestroyImpl.impl_witness.0c1, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.396]
// CHECK:STDOUT:   %.loc10_60.6: %facet_type = converted constants.%EmptyRange.7d6, %facet_value.loc10_60.2 [concrete = constants.%facet_value.396]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_60: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc10_60.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.2(%self.param: ref %tuple.type.784) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref bool = tuple_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_61.1: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_61.1: %facet_type = converted bool, %facet_value.loc10_61.1 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_61.2: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_61.2: %facet_type = converted bool, %facet_value.loc10_61.2 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_61.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_61.3: %facet_type = converted bool, %facet_value.loc10_61.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_61.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_61.4: %facet_type = converted bool, %facet_value.loc10_61.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.3(%self.param: ref %struct_type.has_value.value.6f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_61.1: ref %tuple.type.784 = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_61.1: %facet_type = facet_value constants.%tuple.type.784, (constants.%DestroyImpl.impl_witness.bf6, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %.loc10_61.2: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.1 [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %facet_value.loc10_61.2: %facet_type = facet_value constants.%tuple.type.784, (constants.%DestroyImpl.impl_witness.bf6, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %.loc10_61.3: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.2 [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_61.1)
// CHECK:STDOUT:   %.loc10_61.4: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_61.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_61.5: %facet_type = converted bool, %facet_value.loc10_61.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_61.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_61.6: %facet_type = converted bool, %facet_value.loc10_61.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_61.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.4(%self.param: ref %Optional.9ca) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.6d0 = name_ref value, imports.%Core.import_ref.2f6 [concrete = imports.%field_decl.b75]
// CHECK:STDOUT:   %.loc10_61.1: ref %tuple.type.784 = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_61.1: %facet_type = facet_value constants.%tuple.type.784, (constants.%DestroyImpl.impl_witness.bf6, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %.loc10_61.2: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.1 [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %facet_value.loc10_61.2: %facet_type = facet_value constants.%tuple.type.784, (constants.%DestroyImpl.impl_witness.bf6, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   %.loc10_61.3: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.2 [concrete = constants.%facet_value.856]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_61.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.79c = name_ref has_value, imports.%Core.import_ref.5d0 [concrete = imports.%field_decl.897]
// CHECK:STDOUT:   %.loc10_61.4: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_61.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_61.5: %facet_type = converted bool, %facet_value.loc10_61.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_61.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_61.6: %facet_type = converted bool, %facet_value.loc10_61.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_61.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.5(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_60(%self.param: ref %EmptyRange.7d6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt.1ad: %pattern_type.98b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt.29b: %pattern_type.98b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type.748: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %pattern_type.8b4: type = pattern_type %tuple.type.748 [concrete]
// CHECK:STDOUT:   %.d56: %pattern_type.8b4 = tuple_pattern (%a.patt.1ad, %b.patt.29b) [concrete]
// CHECK:STDOUT:   %EmptyRange.type: type = parameterized_class_name_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.b43: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.35c: %EmptyRange.Make.type.b43 = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.391: %tuple.type.693 = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.74a: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.9bd(%T.f84, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.4af: %tuple.type.as.Copy.impl.Op.type.74a = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet.e83: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.aa2: <witness> = impl_witness imports.%Copy.impl_witness_table.e5a, @tuple.type.as.Copy.impl.9bd(%Copy.facet.e83, %Copy.facet.e83) [concrete]
// CHECK:STDOUT:   %Copy.facet.83b: %Copy.type = facet_value %tuple.type.748, (%Copy.impl_witness.aa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.ad6: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.2f8: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.d66: %EmptyRange.Make.type.2f8 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.d66, @EmptyRange.Make(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.afc: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.a1f: type = unbound_element_type %Optional.afc, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.fbd: type = unbound_element_type %Optional.afc, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f7a: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.c2b: %Optional.Get.type.f7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.23a: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f8f: %Optional.HasValue.type.23a = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.e76: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.706: %EmptyRange.as.Iterate.impl.Next.type.e76 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c99: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.f76: %EmptyRange.as.Iterate.impl.NewCursor.type.c99 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.649: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.194: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.367: %EmptyRange.as.Iterate.impl.NewCursor.type.194 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.76b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.835: %EmptyRange.as.Iterate.impl.Next.type.76b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.ad6, (%Iterate.impl_witness.649) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.324: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.568: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.c72: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.324, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.367, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %.ea9: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.568, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.cc1: type = class_type @Optional, @Optional(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.835, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.5a2: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.6c6: %Optional.HasValue.type.5a2 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.cd8: type = fn_type @Optional.Get, @Optional(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Optional.Get.9d7: %Optional.Get.type.cd8 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.c4d: type = unbound_element_type %Optional.cc1, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.260: type = unbound_element_type %Optional.cc1, %tuple.type.748 [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9b1: type = struct_type {.has_value: bool, .value: %tuple.type.748} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.6c6, @Optional.HasValue(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.9d7, @Optional.Get(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %facet_value.34a: %facet_type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.72d: %SubobjectDestroy.type = facet_value %tuple.type.748, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4bb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.72d) [concrete]
// CHECK:STDOUT:   %facet_value.9a2: %facet_type = facet_value %tuple.type.748, (%DestroyImpl.impl_witness.4bb, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.1b5: %facet_type = facet_value bool, (%DestroyImpl.impl_witness.778, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8d6: %SubobjectDestroy.type = facet_value %Optional.cc1, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f64: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8d6) [concrete]
// CHECK:STDOUT:   %facet_value.f0b: %facet_type = facet_value %Optional.cc1, (%DestroyImpl.impl_witness.f64, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6b7: %SubobjectDestroy.type = facet_value %empty_struct_type, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.011: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6b7) [concrete]
// CHECK:STDOUT:   %facet_value.b01: %facet_type = facet_value %empty_struct_type, (%DestroyImpl.impl_witness.011, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_48 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.bf2: %SubobjectDestroy.type = facet_value %EmptyRange.ad6, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.314: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.bf2) [concrete]
// CHECK:STDOUT:   %facet_value.2ed: %facet_type = facet_value %EmptyRange.ad6, (%DestroyImpl.impl_witness.314, %custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.5e8: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.b43) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.35c)]
// CHECK:STDOUT:   %Core.import_ref.08a: @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.74a) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.4af)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5a = impl_witness_table (%Core.import_ref.08a), @tuple.type.as.Copy.impl.9bd [concrete]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5b1: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.23a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f8f)]
// CHECK:STDOUT:   %Core.import_ref.93e: @Optional.%Optional.Get.type (%Optional.Get.type.f7a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.c2b)]
// CHECK:STDOUT:   %Core.import_ref.5d0: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.897]
// CHECK:STDOUT:   %Core.import_ref.2f6: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.b75]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.9b4: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.c99) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.f76)]
// CHECK:STDOUT:   %Main.import_ref.405: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.e76) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.706)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.9b4, %Main.import_ref.405), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.b75: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.897: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.98b = value_binding_pattern a [concrete = constants.%a.patt.1ad]
// CHECK:STDOUT:     %b.patt: %pattern_type.98b = value_binding_pattern b [concrete = constants.%b.patt.29b]
// CHECK:STDOUT:     %.loc10_19: %pattern_type.8b4 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.d56]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyRange.ref: %EmptyRange.type = name_ref EmptyRange, imports.%Main.EmptyRange [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %C.ref.loc10_36: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_39: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %.loc10_40: %tuple.type.693 = tuple_literal (%C.ref.loc10_36, %C.ref.loc10_39) [concrete = constants.%tuple.391]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.748, (constants.%Copy.impl_witness.aa2) [concrete = constants.%Copy.facet.83b]
// CHECK:STDOUT:   %.loc10_41: %Copy.type = converted %.loc10_40, %Copy.facet [concrete = constants.%Copy.facet.83b]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.ad6]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.2f8 = specific_constant imports.%Main.import_ref.5e8, @EmptyRange(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.Make.d66]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.2f8 = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.d66]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.ad6 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.ad6 to %.loc10_48.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.ad6 = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.c72 = impl_witness_access constants.%Iterate.impl_witness.649, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.367]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.ad6 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.ea9 = impl_witness_access constants.%Iterate.impl_witness.649, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.835]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.cc1 = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.ad6 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.cc1 to %.loc10_49.1 = call %bound_method.loc10_49.4(%.loc10_48.4, %var)
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.cc1 = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.5a2 = specific_constant imports.%Core.import_ref.5b1, @Optional(constants.%Copy.facet.83b) [concrete = constants.%Optional.HasValue.6c6]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.5a2 = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.6c6]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.83b) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.cc1 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.cd8 = specific_constant imports.%Core.import_ref.93e, @Optional(constants.%Copy.facet.83b) [concrete = constants.%Optional.Get.9d7]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.cd8 = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.9d7]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.83b) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.748 = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.cc1 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.Get.call: init %tuple.type.748 to %.loc10_49.8 = call %bound_method.loc10_49.6(%.loc10_49.9)
// CHECK:STDOUT:   %.loc10_49.10: ref %tuple.type.748 = temporary %.loc10_49.8, %Optional.Get.call
// CHECK:STDOUT:   %tuple.elem0: ref %C = tuple_access %.loc10_49.10, element0
// CHECK:STDOUT:   %tuple.elem1: ref %C = tuple_access %.loc10_49.10, element1
// CHECK:STDOUT:   %.loc10_49.11: %C = acquire_value %tuple.elem0
// CHECK:STDOUT:   %C.ref.loc10_12: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %a: %C = wrapper_binding a, %.loc10_49.11
// CHECK:STDOUT:   %.loc10_49.12: %C = acquire_value %tuple.elem1
// CHECK:STDOUT:   %C.ref.loc10_18: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %b: %C = wrapper_binding b, %.loc10_49.12
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %facet_value.loc10_49.1: %facet_type = facet_value constants.%tuple.type.748, (constants.%DestroyImpl.impl_witness.4bb, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %.loc10_49.13: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.1 [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %facet_value.loc10_49.2: %facet_type = facet_value constants.%tuple.type.748, (constants.%DestroyImpl.impl_witness.4bb, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %.loc10_49.14: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.2 [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_49.10)
// CHECK:STDOUT:   %facet_value.loc10_49.3: %facet_type = facet_value constants.%Optional.cc1, (constants.%DestroyImpl.impl_witness.f64, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.f0b]
// CHECK:STDOUT:   %.loc10_49.15: %facet_type = converted constants.%Optional.cc1, %facet_value.loc10_49.3 [concrete = constants.%facet_value.f0b]
// CHECK:STDOUT:   %facet_value.loc10_49.4: %facet_type = facet_value constants.%Optional.cc1, (constants.%DestroyImpl.impl_witness.f64, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.f0b]
// CHECK:STDOUT:   %.loc10_49.16: %facet_type = converted constants.%Optional.cc1, %facet_value.loc10_49.4 [concrete = constants.%facet_value.f0b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_49.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc10_49.5: %facet_type = facet_value constants.%Optional.cc1, (constants.%DestroyImpl.impl_witness.f64, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.f0b]
// CHECK:STDOUT:   %.loc10_49.17: %facet_type = converted constants.%Optional.cc1, %facet_value.loc10_49.5 [concrete = constants.%facet_value.f0b]
// CHECK:STDOUT:   %facet_value.loc10_49.6: %facet_type = facet_value constants.%Optional.cc1, (constants.%DestroyImpl.impl_witness.f64, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.f0b]
// CHECK:STDOUT:   %.loc10_49.18: %facet_type = converted constants.%Optional.cc1, %facet_value.loc10_49.6 [concrete = constants.%facet_value.f0b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_49.2)
// CHECK:STDOUT:   %facet_value.loc10_49.7: %facet_type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %.loc10_49.19: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_49.7 [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %facet_value.loc10_49.8: %facet_type = facet_value constants.%empty_struct_type, (constants.%DestroyImpl.impl_witness.011, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   %.loc10_49.20: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_49.8 [concrete = constants.%facet_value.b01]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   %facet_value.loc10_48.1: %facet_type = facet_value constants.%EmptyRange.ad6, (constants.%DestroyImpl.impl_witness.314, constants.%custom_witness.b80ef6.7) [concrete = constants.%facet_value.2ed]
// CHECK:STDOUT:   %.loc10_48.5: %facet_type = converted constants.%EmptyRange.ad6, %facet_value.loc10_48.1 [concrete = constants.%facet_value.2ed]
// CHECK:STDOUT:   %facet_value.loc10_48.2: %facet_type = facet_value constants.%EmptyRange.ad6, (constants.%DestroyImpl.impl_witness.314, constants.%custom_witness.b80ef6.7) [concrete = constants.%facet_value.2ed]
// CHECK:STDOUT:   %.loc10_48.6: %facet_type = converted constants.%EmptyRange.ad6, %facet_value.loc10_48.2 [concrete = constants.%facet_value.2ed]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_48: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc10_48.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.3(%self.param: ref %tuple.type.748) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %C = tuple_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_49.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_49.1: %facet_type = converted constants.%C, %facet_value.loc10_49.1 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_49.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_49.2: %facet_type = converted constants.%C, %facet_value.loc10_49.2 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %C = tuple_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_49.3: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_49.3: %facet_type = converted constants.%C, %facet_value.loc10_49.3 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc10_49.4: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc10_49.4: %facet_type = converted constants.%C, %facet_value.loc10_49.4 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.4(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.5(%self.param: ref %struct_type.has_value.value.9b1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_49.1: ref %tuple.type.748 = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_49.1: %facet_type = facet_value constants.%tuple.type.748, (constants.%DestroyImpl.impl_witness.4bb, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %.loc10_49.2: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.1 [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %facet_value.loc10_49.2: %facet_type = facet_value constants.%tuple.type.748, (constants.%DestroyImpl.impl_witness.4bb, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %.loc10_49.3: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.2 [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_49.1)
// CHECK:STDOUT:   %.loc10_49.4: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_49.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_49.5: %facet_type = converted bool, %facet_value.loc10_49.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_49.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_49.6: %facet_type = converted bool, %facet_value.loc10_49.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_49.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.6(%self.param: ref %Optional.cc1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.260 = name_ref value, imports.%Core.import_ref.2f6 [concrete = imports.%field_decl.b75]
// CHECK:STDOUT:   %.loc10_49.1: ref %tuple.type.748 = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_49.1: %facet_type = facet_value constants.%tuple.type.748, (constants.%DestroyImpl.impl_witness.4bb, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %.loc10_49.2: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.1 [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %facet_value.loc10_49.2: %facet_type = facet_value constants.%tuple.type.748, (constants.%DestroyImpl.impl_witness.4bb, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   %.loc10_49.3: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.2 [concrete = constants.%facet_value.9a2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_49.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.c4d = name_ref has_value, imports.%Core.import_ref.5d0 [concrete = imports.%field_decl.897]
// CHECK:STDOUT:   %.loc10_49.4: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_49.3: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_49.5: %facet_type = converted bool, %facet_value.loc10_49.3 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %facet_value.loc10_49.4: %facet_type = facet_value bool, (constants.%DestroyImpl.impl_witness.778, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   %.loc10_49.6: %facet_type = converted bool, %facet_value.loc10_49.4 [concrete = constants.%facet_value.1b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_49.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_48(%self.param: ref %EmptyRange.ad6) = "no_op";
// CHECK:STDOUT:
