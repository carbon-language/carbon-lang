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
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+7]]:7: error: cannot implicitly convert expression of type `C` to `X` [ConversionFailure]
  // CHECK:STDERR:   for (unused x: X in EmptyRange(C).Make()) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+4]]:7: note: type `C` does not implement interface `Core.ImplicitAs(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   for (unused x: X in EmptyRange(C).Make()) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %EmptyRange.Make.type.ecd: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.764: %EmptyRange.Make.type.ecd = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %EmptyRange.842: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.647: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.e1f: %EmptyRange.Make.type.647 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.e1f, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.b41: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.7e8: type = unbound_element_type %Optional.b41, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.3da: type = unbound_element_type %Optional.b41, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.ae7: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.df6: %Optional.Get.type.ae7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e50: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.8d1: %Optional.HasValue.type.e50 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.89f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.3bc: %EmptyRange.as.Iterate.impl.Next.type.89f = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.025: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.a6f: %EmptyRange.as.Iterate.impl.NewCursor.type.025 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.282: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.695: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.995: %EmptyRange.as.Iterate.impl.NewCursor.type.695 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.ab9: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.16c: %EmptyRange.as.Iterate.impl.Next.type.ab9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.842, (%Iterate.impl_witness.282) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.9f0: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.270: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.2cd: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.9f0, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.995, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.270, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.75f: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.16c, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.907: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ede: %Optional.HasValue.type.907 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d61: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.fd5: %Optional.Get.type.d61 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.96e: type = unbound_element_type %Optional.75f, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.55d: type = unbound_element_type %Optional.75f, %C [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9e7: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.ede, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.fd5, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.1: type = fn_type @DestroyImpl.WithSelf.Op.loc10_36.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.1: %DestroyImpl.WithSelf.Op.type.b77747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.1: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.1), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc10_36.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_36.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_36.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.f6a: %facet_type = facet_value %C, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc10_36.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.5: type = fn_type @DestroyImpl.WithSelf.Op.loc10_36.5 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.5: %DestroyImpl.WithSelf.Op.type.b77747.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.5), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_36.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.1f4: %facet_type = facet_value bool, (%custom_witness.2cb555.3, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_36.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.9a4: %facet_type = facet_value %Optional.75f, (%custom_witness.2cb555.5, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %facet_value.ab3: %facet_type = facet_value %empty_struct_type, (%custom_witness.2cb555.1, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc10_35 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.6: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_35 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.8d3: %facet_type = facet_value %EmptyRange.842, (%custom_witness.2cb555.6, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.bfc: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.ecd) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.764)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e50) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.8d1)]
// CHECK:STDOUT:   %Core.import_ref.615: @Optional.%Optional.Get.type (%Optional.Get.type.ae7) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.df6)]
// CHECK:STDOUT:   %Core.import_ref.b24: @Optional.%Optional.elem.1 (%Optional.elem.3da) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a4]
// CHECK:STDOUT:   %Core.import_ref.147: @Optional.%Optional.elem.2 (%Optional.elem.7e8) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.210]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.626: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.025) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.a6f)]
// CHECK:STDOUT:   %Main.import_ref.ac5: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.89f) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.3bc)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.626, %Main.import_ref.ac5), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.210: @Optional.%Optional.elem.2 (%Optional.elem.7e8) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a4: @Optional.%Optional.elem.1 (%Optional.elem.3da) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
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
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.842]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.647 = specific_constant imports.%Main.import_ref.bfc, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.e1f]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.647 = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.e1f]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.842 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.842 to %.loc10_35.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.842 = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.2cd = impl_witness_access constants.%Iterate.impl_witness.282, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.995]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.842 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.a6f = impl_witness_access constants.%Iterate.impl_witness.282, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.16c]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.75f = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.842 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.75f to %.loc10_36.1 = call %bound_method.loc10_36.4(%.loc10_35.4, %var)
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.75f = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.907 = specific_constant imports.%Core.import_ref.3d4, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.ede]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.907 = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.ede]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.75f = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.d61 = specific_constant imports.%Core.import_ref.615, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.fd5]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.d61 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.fd5]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.75f = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_36.8 = call %bound_method.loc10_36.6(%.loc10_36.9)
// CHECK:STDOUT:   %.loc10_36.10: ref %C = temporary %.loc10_36.8, %Optional.Get.call
// CHECK:STDOUT:   %.loc10_36.11: %C = acquire_value %.loc10_36.10
// CHECK:STDOUT:   %C.ref.loc10_11: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc10_36.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%c.ref)
// CHECK:STDOUT:   %facet_value.loc10_36.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_36.12: %facet_type = converted constants.%C, %facet_value.loc10_36.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_36.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_36.13: %facet_type = converted constants.%C, %facet_value.loc10_36.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_36.10)
// CHECK:STDOUT:   %facet_value.loc10_36.3: %facet_type = facet_value constants.%Optional.75f, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %.loc10_36.14: %facet_type = converted constants.%Optional.75f, %facet_value.loc10_36.3 [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %facet_value.loc10_36.4: %facet_type = facet_value constants.%Optional.75f, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %.loc10_36.15: %facet_type = converted constants.%Optional.75f, %facet_value.loc10_36.4 [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_36.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc10_36.5: %facet_type = facet_value constants.%Optional.75f, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %.loc10_36.16: %facet_type = converted constants.%Optional.75f, %facet_value.loc10_36.5 [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %facet_value.loc10_36.6: %facet_type = facet_value constants.%Optional.75f, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %.loc10_36.17: %facet_type = converted constants.%Optional.75f, %facet_value.loc10_36.6 [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_36.2)
// CHECK:STDOUT:   %facet_value.loc10_36.7: %facet_type = facet_value constants.%empty_struct_type, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %.loc10_36.18: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_36.7 [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %facet_value.loc10_36.8: %facet_type = facet_value constants.%empty_struct_type, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %.loc10_36.19: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_36.8 [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   %facet_value.loc10_35.1: %facet_type = facet_value constants.%EmptyRange.842, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.8d3]
// CHECK:STDOUT:   %.loc10_35.5: %facet_type = converted constants.%EmptyRange.842, %facet_value.loc10_35.1 [concrete = constants.%facet_value.8d3]
// CHECK:STDOUT:   %facet_value.loc10_35.2: %facet_type = facet_value constants.%EmptyRange.842, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.8d3]
// CHECK:STDOUT:   %.loc10_35.6: %facet_type = converted constants.%EmptyRange.842, %facet_value.loc10_35.2 [concrete = constants.%facet_value.8d3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_35: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc10_35.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_36.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_36.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_36.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_36.4(%self.param: ref %struct_type.has_value.value.9e7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_36.5(%self.param: ref %Optional.75f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_36.1: ref %C = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_36.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_36.2: %facet_type = converted constants.%C, %facet_value.loc10_36.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_36.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_36.3: %facet_type = converted constants.%C, %facet_value.loc10_36.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_36.1)
// CHECK:STDOUT:   %.loc10_36.4: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_36.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_36.5: %facet_type = converted bool, %facet_value.loc10_36.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_36.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_36.6: %facet_type = converted bool, %facet_value.loc10_36.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_36.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_36.5(%self.param: ref %Optional.75f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.55d = name_ref value, imports.%Core.import_ref.147 [concrete = imports.%field_decl.210]
// CHECK:STDOUT:   %.loc10_36.1: ref %C = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_36.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_36.2: %facet_type = converted constants.%C, %facet_value.loc10_36.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_36.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_36.3: %facet_type = converted constants.%C, %facet_value.loc10_36.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_36.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.96e = name_ref has_value, imports.%Core.import_ref.b24 [concrete = imports.%field_decl.6a4]
// CHECK:STDOUT:   %.loc10_36.4: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_36.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_36.5: %facet_type = converted bool, %facet_value.loc10_36.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_36.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_36.6: %facet_type = converted bool, %facet_value.loc10_36.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_36.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_35(%self.param: ref %EmptyRange.842) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_35(%self.param: ref %EmptyRange.842) = "no_op";
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
// CHECK:STDOUT:   %EmptyRange.Make.type.ecd: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.764: %EmptyRange.Make.type.ecd = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %EmptyRange.842: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.647: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.e1f: %EmptyRange.Make.type.647 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.e1f, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.b41: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.7e8: type = unbound_element_type %Optional.b41, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.3da: type = unbound_element_type %Optional.b41, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.ae7: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.df6: %Optional.Get.type.ae7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e50: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.8d1: %Optional.HasValue.type.e50 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.89f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.3bc: %EmptyRange.as.Iterate.impl.Next.type.89f = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.025: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.a6f: %EmptyRange.as.Iterate.impl.NewCursor.type.025 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.282: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.695: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.995: %EmptyRange.as.Iterate.impl.NewCursor.type.695 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.ab9: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.16c: %EmptyRange.as.Iterate.impl.Next.type.ab9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.842, (%Iterate.impl_witness.282) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.9f0: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.270: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.2cd: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.9f0, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.995, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.270, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.75f: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.16c, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.907: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ede: %Optional.HasValue.type.907 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d61: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.fd5: %Optional.Get.type.d61 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.96e: type = unbound_element_type %Optional.75f, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.55d: type = unbound_element_type %Optional.75f, %C [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9e7: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.ede, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.fd5, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.1: type = fn_type @DestroyImpl.WithSelf.Op.loc10_40.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.1: %DestroyImpl.WithSelf.Op.type.b77747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.1: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.1), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc10_40.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc10_40.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.5: type = fn_type @DestroyImpl.WithSelf.Op.loc10_40.5 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.5: %DestroyImpl.WithSelf.Op.type.b77747.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.5), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_40.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_40.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_40.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.f6a: %facet_type = facet_value %C, (%custom_witness.2cb555.3, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %facet_value.1f4: %facet_type = facet_value bool, (%custom_witness.2cb555.1, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_40.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.9a4: %facet_type = facet_value %Optional.75f, (%custom_witness.2cb555.5, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %facet_value.ab3: %facet_type = facet_value %empty_struct_type, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc10_39 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.6: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_39 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.8d3: %facet_type = facet_value %EmptyRange.842, (%custom_witness.2cb555.6, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.bfc: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.ecd) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.764)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e50) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.8d1)]
// CHECK:STDOUT:   %Core.import_ref.615: @Optional.%Optional.Get.type (%Optional.Get.type.ae7) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.df6)]
// CHECK:STDOUT:   %Core.import_ref.b24: @Optional.%Optional.elem.1 (%Optional.elem.3da) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a4]
// CHECK:STDOUT:   %Core.import_ref.147: @Optional.%Optional.elem.2 (%Optional.elem.7e8) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.210]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.626: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.025) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.a6f)]
// CHECK:STDOUT:   %Main.import_ref.ac5: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.89f) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.3bc)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.626, %Main.import_ref.ac5), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.210: @Optional.%Optional.elem.2 (%Optional.elem.7e8) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a4: @Optional.%Optional.elem.1 (%Optional.elem.3da) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
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
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.842]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.647 = specific_constant imports.%Main.import_ref.bfc, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.e1f]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.647 = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.e1f]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.842 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.842 to %.loc10_39.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.842 = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.2cd = impl_witness_access constants.%Iterate.impl_witness.282, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.995]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.842 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.a6f = impl_witness_access constants.%Iterate.impl_witness.282, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.16c]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.75f = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.842 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.75f to %.loc10_40.1 = call %bound_method.loc10_40.4(%.loc10_39.4, %var)
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.75f = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.907 = specific_constant imports.%Core.import_ref.3d4, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.ede]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.907 = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.ede]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.75f = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.d61 = specific_constant imports.%Core.import_ref.615, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.fd5]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.d61 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.fd5]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8.1: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.75f = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_8.1 = call %bound_method.loc10_40.6(%.loc10_40.8)
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr: %ptr.6b6 = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr)
// CHECK:STDOUT:   %facet_value.loc10_40.1: %facet_type = facet_value constants.%Optional.75f, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %.loc10_40.9: %facet_type = converted constants.%Optional.75f, %facet_value.loc10_40.1 [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %facet_value.loc10_40.2: %facet_type = facet_value constants.%Optional.75f, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %.loc10_40.10: %facet_type = converted constants.%Optional.75f, %facet_value.loc10_40.2 [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_40.2)
// CHECK:STDOUT:   %facet_value.loc10_8.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_8.2: %facet_type = converted constants.%C, %facet_value.loc10_8.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_8.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_8.3: %facet_type = converted constants.%C, %facet_value.loc10_8.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_8.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%c.var)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc10_40.3: %facet_type = facet_value constants.%Optional.75f, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %.loc10_40.11: %facet_type = converted constants.%Optional.75f, %facet_value.loc10_40.3 [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %facet_value.loc10_40.4: %facet_type = facet_value constants.%Optional.75f, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   %.loc10_40.12: %facet_type = converted constants.%Optional.75f, %facet_value.loc10_40.4 [concrete = constants.%facet_value.9a4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_40.2)
// CHECK:STDOUT:   %facet_value.loc10_8.3: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_8.4: %facet_type = converted constants.%C, %facet_value.loc10_8.3 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_8.4: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_8.5: %facet_type = converted constants.%C, %facet_value.loc10_8.4 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_8.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%c.var)
// CHECK:STDOUT:   %facet_value.loc10_40.5: %facet_type = facet_value constants.%empty_struct_type, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %.loc10_40.13: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_40.5 [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %facet_value.loc10_40.6: %facet_type = facet_value constants.%empty_struct_type, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %.loc10_40.14: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_40.6 [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%var)
// CHECK:STDOUT:   %facet_value.loc10_39.1: %facet_type = facet_value constants.%EmptyRange.842, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.8d3]
// CHECK:STDOUT:   %.loc10_39.5: %facet_type = converted constants.%EmptyRange.842, %facet_value.loc10_39.1 [concrete = constants.%facet_value.8d3]
// CHECK:STDOUT:   %facet_value.loc10_39.2: %facet_type = facet_value constants.%EmptyRange.842, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.8d3]
// CHECK:STDOUT:   %.loc10_39.6: %facet_type = converted constants.%EmptyRange.842, %facet_value.loc10_39.2 [concrete = constants.%facet_value.8d3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_39: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc10_39.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_40.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_40.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_40.3(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_40.4(%self.param: ref %struct_type.has_value.value.9e7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_40.5(%self.param: ref %Optional.75f) = "no_op";
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
// CHECK:STDOUT:   %facet_value.loc10_40.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_40.2: %facet_type = converted constants.%C, %facet_value.loc10_40.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_40.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_40.3: %facet_type = converted constants.%C, %facet_value.loc10_40.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_40.1)
// CHECK:STDOUT:   %.loc10_40.4: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_40.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_40.5: %facet_type = converted bool, %facet_value.loc10_40.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_40.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_40.6: %facet_type = converted bool, %facet_value.loc10_40.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_40.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_40.5(%self.param: ref %Optional.75f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.55d = name_ref value, imports.%Core.import_ref.147 [concrete = imports.%field_decl.210]
// CHECK:STDOUT:   %.loc10_40.1: ref %C = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_40.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_40.2: %facet_type = converted constants.%C, %facet_value.loc10_40.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_40.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_40.3: %facet_type = converted constants.%C, %facet_value.loc10_40.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_40.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.96e = name_ref has_value, imports.%Core.import_ref.b24 [concrete = imports.%field_decl.6a4]
// CHECK:STDOUT:   %.loc10_40.4: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_40.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_40.5: %facet_type = converted bool, %facet_value.loc10_40.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_40.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_40.6: %facet_type = converted bool, %facet_value.loc10_40.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_40.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_39(%self.param: ref %EmptyRange.842) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_39(%self.param: ref %EmptyRange.842) = "no_op";
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
// CHECK:STDOUT:   %EmptyRange.Make.type.ecd: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.764: %EmptyRange.Make.type.ecd = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.52b: %tuple.type.693 = tuple_value (bool, bool) [concrete]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.3ed: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.9bd(%T.f84, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.0f8: %tuple.type.as.Copy.impl.Op.type.3ed = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet.7af: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.493: <witness> = impl_witness imports.%Copy.impl_witness_table.e5a, @tuple.type.as.Copy.impl.9bd(%Copy.facet.7af, %Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Copy.facet.701: %Copy.type = facet_value %tuple.type.784, (%Copy.impl_witness.493) [concrete]
// CHECK:STDOUT:   %EmptyRange.b25: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.d18: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.3a8: %EmptyRange.Make.type.d18 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.3a8, @EmptyRange.Make(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.b41: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.7e8: type = unbound_element_type %Optional.b41, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.3da: type = unbound_element_type %Optional.b41, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.ae7: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.df6: %Optional.Get.type.ae7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e50: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.8d1: %Optional.HasValue.type.e50 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.89f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.3bc: %EmptyRange.as.Iterate.impl.Next.type.89f = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.025: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.a6f: %EmptyRange.as.Iterate.impl.NewCursor.type.025 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.d9b: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.31a: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.778: %EmptyRange.as.Iterate.impl.NewCursor.type.31a = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.250: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.401: %EmptyRange.as.Iterate.impl.Next.type.250 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.b25, (%Iterate.impl_witness.d9b) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.506: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.000: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.cae: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.506, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.778, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %.95a: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.000, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.dbb: type = class_type @Optional, @Optional(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.401, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.9a1: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.fe9: %Optional.HasValue.type.9a1 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f70: type = fn_type @Optional.Get, @Optional(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Optional.Get.e91: %Optional.Get.type.f70 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.84b: type = unbound_element_type %Optional.dbb, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.34d: type = unbound_element_type %Optional.dbb, %tuple.type.784 [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.6f9: type = struct_type {.has_value: bool, .value: %tuple.type.784} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.fe9, @Optional.HasValue(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.e91, @Optional.Get(%Copy.facet.701) [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.1: type = fn_type @DestroyImpl.WithSelf.Op.loc10_61.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.1: %DestroyImpl.WithSelf.Op.type.b77747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.1: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.1), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc10_61.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_61.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_61.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %facet_value.1f4: %facet_type = facet_value bool, (%custom_witness.2cb555.1, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.645: %facet_type = facet_value %tuple.type.784, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc10_61.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.4: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_61.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.534: %facet_type = facet_value %Optional.dbb, (%custom_witness.2cb555.4, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.5: type = fn_type @DestroyImpl.WithSelf.Op.loc10_61.5 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.5: %DestroyImpl.WithSelf.Op.type.b77747.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.5), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_61.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.ab3: %facet_type = facet_value %empty_struct_type, (%custom_witness.2cb555.5, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc10_60 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.6: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_60 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.a42: %facet_type = facet_value %EmptyRange.b25, (%custom_witness.2cb555.6, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.bfc: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.ecd) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.764)]
// CHECK:STDOUT:   %Core.import_ref.44a: @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.3ed) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.0f8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5a = impl_witness_table (%Core.import_ref.44a), @tuple.type.as.Copy.impl.9bd [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e50) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.8d1)]
// CHECK:STDOUT:   %Core.import_ref.615: @Optional.%Optional.Get.type (%Optional.Get.type.ae7) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.df6)]
// CHECK:STDOUT:   %Core.import_ref.b24: @Optional.%Optional.elem.1 (%Optional.elem.3da) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a4]
// CHECK:STDOUT:   %Core.import_ref.147: @Optional.%Optional.elem.2 (%Optional.elem.7e8) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.210]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.626: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.025) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.a6f)]
// CHECK:STDOUT:   %Main.import_ref.ac5: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.89f) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.3bc)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.626, %Main.import_ref.ac5), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.210: @Optional.%Optional.elem.2 (%Optional.elem.7e8) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d0 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a4: @Optional.%Optional.elem.1 (%Optional.elem.3da) = field_decl has_value, element0, %Core.import_ref.b0d0 in [concrete] {}
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
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.784, (constants.%Copy.impl_witness.493) [concrete = constants.%Copy.facet.701]
// CHECK:STDOUT:   %.loc10_53: %Copy.type = converted %.loc10_52, %Copy.facet [concrete = constants.%Copy.facet.701]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.b25]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.d18 = specific_constant imports.%Main.import_ref.bfc, @EmptyRange(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.Make.3a8]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.d18 = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.3a8]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.b25 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.b25 to %.loc10_60.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.b25 = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.cae = impl_witness_access constants.%Iterate.impl_witness.d9b, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.778]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.b25 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.95a = impl_witness_access constants.%Iterate.impl_witness.d9b, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.401]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.701) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.dbb = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.b25 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.dbb to %.loc10_61.1 = call %bound_method.loc10_61.4(%.loc10_60.4, %var)
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.dbb = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.9a1 = specific_constant imports.%Core.import_ref.3d4, @Optional(constants.%Copy.facet.701) [concrete = constants.%Optional.HasValue.fe9]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.9a1 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.fe9]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.701) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.dbb = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.f70 = specific_constant imports.%Core.import_ref.615, @Optional(constants.%Copy.facet.701) [concrete = constants.%Optional.Get.e91]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f70 = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.e91]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.701) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.dbb = acquire_value %.loc10_61.2
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
// CHECK:STDOUT:   %facet_value.loc10_61.1: %facet_type = facet_value constants.%tuple.type.784, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %.loc10_61.13: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.1 [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %facet_value.loc10_61.2: %facet_type = facet_value constants.%tuple.type.784, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %.loc10_61.14: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.2 [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_61.10)
// CHECK:STDOUT:   %facet_value.loc10_61.3: %facet_type = facet_value constants.%Optional.dbb, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.534]
// CHECK:STDOUT:   %.loc10_61.15: %facet_type = converted constants.%Optional.dbb, %facet_value.loc10_61.3 [concrete = constants.%facet_value.534]
// CHECK:STDOUT:   %facet_value.loc10_61.4: %facet_type = facet_value constants.%Optional.dbb, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.534]
// CHECK:STDOUT:   %.loc10_61.16: %facet_type = converted constants.%Optional.dbb, %facet_value.loc10_61.4 [concrete = constants.%facet_value.534]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_61.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc10_61.5: %facet_type = facet_value constants.%Optional.dbb, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.534]
// CHECK:STDOUT:   %.loc10_61.17: %facet_type = converted constants.%Optional.dbb, %facet_value.loc10_61.5 [concrete = constants.%facet_value.534]
// CHECK:STDOUT:   %facet_value.loc10_61.6: %facet_type = facet_value constants.%Optional.dbb, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.534]
// CHECK:STDOUT:   %.loc10_61.18: %facet_type = converted constants.%Optional.dbb, %facet_value.loc10_61.6 [concrete = constants.%facet_value.534]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_61.2)
// CHECK:STDOUT:   %facet_value.loc10_61.7: %facet_type = facet_value constants.%empty_struct_type, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %.loc10_61.19: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_61.7 [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %facet_value.loc10_61.8: %facet_type = facet_value constants.%empty_struct_type, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %.loc10_61.20: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_61.8 [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   %facet_value.loc10_60.1: %facet_type = facet_value constants.%EmptyRange.b25, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.a42]
// CHECK:STDOUT:   %.loc10_60.5: %facet_type = converted constants.%EmptyRange.b25, %facet_value.loc10_60.1 [concrete = constants.%facet_value.a42]
// CHECK:STDOUT:   %facet_value.loc10_60.2: %facet_type = facet_value constants.%EmptyRange.b25, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.a42]
// CHECK:STDOUT:   %.loc10_60.6: %facet_type = converted constants.%EmptyRange.b25, %facet_value.loc10_60.2 [concrete = constants.%facet_value.a42]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_60: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc10_60.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_61.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_61.2(%self.param: ref %tuple.type.784) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.2(%self.param: ref %tuple.type.784) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref bool = tuple_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_61.1: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_61.1: %facet_type = converted bool, %facet_value.loc10_61.1 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_61.2: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_61.2: %facet_type = converted bool, %facet_value.loc10_61.2 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_61.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_61.3: %facet_type = converted bool, %facet_value.loc10_61.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_61.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_61.4: %facet_type = converted bool, %facet_value.loc10_61.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_61.3(%self.param: ref %struct_type.has_value.value.6f9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_61.4(%self.param: ref %Optional.dbb) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.3(%self.param: ref %struct_type.has_value.value.6f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_61.1: ref %tuple.type.784 = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_61.1: %facet_type = facet_value constants.%tuple.type.784, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %.loc10_61.2: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.1 [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %facet_value.loc10_61.2: %facet_type = facet_value constants.%tuple.type.784, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %.loc10_61.3: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.2 [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_61.1)
// CHECK:STDOUT:   %.loc10_61.4: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_61.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_61.5: %facet_type = converted bool, %facet_value.loc10_61.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_61.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_61.6: %facet_type = converted bool, %facet_value.loc10_61.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_61.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.4(%self.param: ref %Optional.dbb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.34d = name_ref value, imports.%Core.import_ref.147 [concrete = imports.%field_decl.210]
// CHECK:STDOUT:   %.loc10_61.1: ref %tuple.type.784 = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_61.1: %facet_type = facet_value constants.%tuple.type.784, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %.loc10_61.2: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.1 [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %facet_value.loc10_61.2: %facet_type = facet_value constants.%tuple.type.784, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   %.loc10_61.3: %facet_type = converted constants.%tuple.type.784, %facet_value.loc10_61.2 [concrete = constants.%facet_value.645]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_61.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.84b = name_ref has_value, imports.%Core.import_ref.b24 [concrete = imports.%field_decl.6a4]
// CHECK:STDOUT:   %.loc10_61.4: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_61.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_61.5: %facet_type = converted bool, %facet_value.loc10_61.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_61.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_61.6: %facet_type = converted bool, %facet_value.loc10_61.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_61.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_61.5(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_61.5(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_60(%self.param: ref %EmptyRange.b25) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_60(%self.param: ref %EmptyRange.b25) = "no_op";
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
// CHECK:STDOUT:   %EmptyRange.Make.type.ecd: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.764: %EmptyRange.Make.type.ecd = struct_value () [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.391: %tuple.type.693 = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %U: %Copy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.type.3ed: type = fn_type @tuple.type.as.Copy.impl.Op.1, @tuple.type.as.Copy.impl.9bd(%T.f84, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.as.Copy.impl.Op.0f8: %tuple.type.as.Copy.impl.Op.type.3ed = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet.e83: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5b5: <witness> = impl_witness imports.%Copy.impl_witness_table.e5a, @tuple.type.as.Copy.impl.9bd(%Copy.facet.e83, %Copy.facet.e83) [concrete]
// CHECK:STDOUT:   %Copy.facet.38c: %Copy.type = facet_value %tuple.type.748, (%Copy.impl_witness.5b5) [concrete]
// CHECK:STDOUT:   %EmptyRange.f31: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.300: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.d9d: %EmptyRange.Make.type.300 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.d9d, @EmptyRange.Make(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.b41: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.7e8: type = unbound_element_type %Optional.b41, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.3da: type = unbound_element_type %Optional.b41, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.ae7: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.df6: %Optional.Get.type.ae7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e50: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.8d1: %Optional.HasValue.type.e50 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.89f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.3bc: %EmptyRange.as.Iterate.impl.Next.type.89f = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.025: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.a6f: %EmptyRange.as.Iterate.impl.NewCursor.type.025 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.impl_witness.8c7: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.906: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.746: %EmptyRange.as.Iterate.impl.NewCursor.type.906 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.4af: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.582: %EmptyRange.as.Iterate.impl.Next.type.4af = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.f31, (%Iterate.impl_witness.8c7) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.499: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.cea: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.890: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.499, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.746, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %.97c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.cea, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.87c: type = class_type @Optional, @Optional(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.582, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.318: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ffc: %Optional.HasValue.type.318 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f74: type = fn_type @Optional.Get, @Optional(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Optional.Get.af9: %Optional.Get.type.f74 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.ec5: type = unbound_element_type %Optional.87c, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.a48: type = unbound_element_type %Optional.87c, %tuple.type.748 [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9b1: type = struct_type {.has_value: bool, .value: %tuple.type.748} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.ffc, @Optional.HasValue(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.af9, @Optional.Get(%Copy.facet.38c) [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.1: type = fn_type @DestroyImpl.WithSelf.Op.loc10_49.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.1: %DestroyImpl.WithSelf.Op.type.b77747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.1: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.1), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc10_49.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc10_49.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %facet_value.f6a: %facet_type = facet_value %C, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.e9d: %facet_type = facet_value %tuple.type.748, (%custom_witness.2cb555.3, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc10_49.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.4: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc10_49.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.6: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.1f4: %facet_type = facet_value bool, (%custom_witness.2cb555.4, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_49.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.c74: %facet_type = facet_value %Optional.87c, (%custom_witness.2cb555.6, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %facet_value.ab3: %facet_type = facet_value %empty_struct_type, (%custom_witness.2cb555.1, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.7: type = fn_type @DestroyImpl.WithSelf.Op.loc10_48 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.7: %DestroyImpl.WithSelf.Op.type.b77747.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.7: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.7), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_48 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.850: %facet_type = facet_value %EmptyRange.f31, (%custom_witness.2cb555.7, %custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.bfc: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.ecd) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.764)]
// CHECK:STDOUT:   %Core.import_ref.44a: @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.3ed) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.0f8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5a = impl_witness_table (%Core.import_ref.44a), @tuple.type.as.Copy.impl.9bd [concrete]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e50) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.8d1)]
// CHECK:STDOUT:   %Core.import_ref.615: @Optional.%Optional.Get.type (%Optional.Get.type.ae7) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.df6)]
// CHECK:STDOUT:   %Core.import_ref.b24: @Optional.%Optional.elem.1 (%Optional.elem.3da) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a4]
// CHECK:STDOUT:   %Core.import_ref.147: @Optional.%Optional.elem.2 (%Optional.elem.7e8) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.210]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.626: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.025) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.a6f)]
// CHECK:STDOUT:   %Main.import_ref.ac5: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.89f) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.3bc)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.626, %Main.import_ref.ac5), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.210: @Optional.%Optional.elem.2 (%Optional.elem.7e8) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a4: @Optional.%Optional.elem.1 (%Optional.elem.3da) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
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
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%tuple.type.748, (constants.%Copy.impl_witness.5b5) [concrete = constants.%Copy.facet.38c]
// CHECK:STDOUT:   %.loc10_41: %Copy.type = converted %.loc10_40, %Copy.facet [concrete = constants.%Copy.facet.38c]
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.f31]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.300 = specific_constant imports.%Main.import_ref.bfc, @EmptyRange(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.Make.d9d]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.300 = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.d9d]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.f31 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.f31 to %.loc10_48.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.f31 = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.890 = impl_witness_access constants.%Iterate.impl_witness.8c7, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.746]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.f31 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.97c = impl_witness_access constants.%Iterate.impl_witness.8c7, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.582]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.38c) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.87c = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.f31 = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.87c to %.loc10_49.1 = call %bound_method.loc10_49.4(%.loc10_48.4, %var)
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.87c = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.318 = specific_constant imports.%Core.import_ref.3d4, @Optional(constants.%Copy.facet.38c) [concrete = constants.%Optional.HasValue.ffc]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.318 = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.ffc]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.38c) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.87c = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.f74 = specific_constant imports.%Core.import_ref.615, @Optional(constants.%Copy.facet.38c) [concrete = constants.%Optional.Get.af9]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f74 = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.af9]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.38c) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.748 = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.87c = acquire_value %.loc10_49.2
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
// CHECK:STDOUT:   %facet_value.loc10_49.1: %facet_type = facet_value constants.%tuple.type.748, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %.loc10_49.13: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.1 [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %facet_value.loc10_49.2: %facet_type = facet_value constants.%tuple.type.748, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %.loc10_49.14: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.2 [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_49.10)
// CHECK:STDOUT:   %facet_value.loc10_49.3: %facet_type = facet_value constants.%Optional.87c, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.c74]
// CHECK:STDOUT:   %.loc10_49.15: %facet_type = converted constants.%Optional.87c, %facet_value.loc10_49.3 [concrete = constants.%facet_value.c74]
// CHECK:STDOUT:   %facet_value.loc10_49.4: %facet_type = facet_value constants.%Optional.87c, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.c74]
// CHECK:STDOUT:   %.loc10_49.16: %facet_type = converted constants.%Optional.87c, %facet_value.loc10_49.4 [concrete = constants.%facet_value.c74]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_49.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc10_49.5: %facet_type = facet_value constants.%Optional.87c, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.c74]
// CHECK:STDOUT:   %.loc10_49.17: %facet_type = converted constants.%Optional.87c, %facet_value.loc10_49.5 [concrete = constants.%facet_value.c74]
// CHECK:STDOUT:   %facet_value.loc10_49.6: %facet_type = facet_value constants.%Optional.87c, (constants.%custom_witness.2cb555.6, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.c74]
// CHECK:STDOUT:   %.loc10_49.18: %facet_type = converted constants.%Optional.87c, %facet_value.loc10_49.6 [concrete = constants.%facet_value.c74]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_49.2)
// CHECK:STDOUT:   %facet_value.loc10_49.7: %facet_type = facet_value constants.%empty_struct_type, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %.loc10_49.19: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_49.7 [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %facet_value.loc10_49.8: %facet_type = facet_value constants.%empty_struct_type, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   %.loc10_49.20: %facet_type = converted constants.%empty_struct_type, %facet_value.loc10_49.8 [concrete = constants.%facet_value.ab3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   %facet_value.loc10_48.1: %facet_type = facet_value constants.%EmptyRange.f31, (constants.%custom_witness.2cb555.7, constants.%custom_witness.b80ef6.7) [concrete = constants.%facet_value.850]
// CHECK:STDOUT:   %.loc10_48.5: %facet_type = converted constants.%EmptyRange.f31, %facet_value.loc10_48.1 [concrete = constants.%facet_value.850]
// CHECK:STDOUT:   %facet_value.loc10_48.2: %facet_type = facet_value constants.%EmptyRange.f31, (constants.%custom_witness.2cb555.7, constants.%custom_witness.b80ef6.7) [concrete = constants.%facet_value.850]
// CHECK:STDOUT:   %.loc10_48.6: %facet_type = converted constants.%EmptyRange.f31, %facet_value.loc10_48.2 [concrete = constants.%facet_value.850]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_48: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc10_48.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_49.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_49.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_49.3(%self.param: ref %tuple.type.748) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.3(%self.param: ref %tuple.type.748) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %C = tuple_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_49.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_49.1: %facet_type = converted constants.%C, %facet_value.loc10_49.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_49.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_49.2: %facet_type = converted constants.%C, %facet_value.loc10_49.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %C = tuple_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_49.3: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_49.3: %facet_type = converted constants.%C, %facet_value.loc10_49.3 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc10_49.4: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc10_49.4: %facet_type = converted constants.%C, %facet_value.loc10_49.4 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_49.4(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_49.5(%self.param: ref %struct_type.has_value.value.9b1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_49.6(%self.param: ref %Optional.87c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.4(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.5(%self.param: ref %struct_type.has_value.value.9b1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_49.1: ref %tuple.type.748 = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_49.1: %facet_type = facet_value constants.%tuple.type.748, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %.loc10_49.2: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.1 [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %facet_value.loc10_49.2: %facet_type = facet_value constants.%tuple.type.748, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %.loc10_49.3: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.2 [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_49.1)
// CHECK:STDOUT:   %.loc10_49.4: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_49.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_49.5: %facet_type = converted bool, %facet_value.loc10_49.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_49.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_49.6: %facet_type = converted bool, %facet_value.loc10_49.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_49.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_49.6(%self.param: ref %Optional.87c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.a48 = name_ref value, imports.%Core.import_ref.147 [concrete = imports.%field_decl.210]
// CHECK:STDOUT:   %.loc10_49.1: ref %tuple.type.748 = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc10_49.1: %facet_type = facet_value constants.%tuple.type.748, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %.loc10_49.2: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.1 [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %facet_value.loc10_49.2: %facet_type = facet_value constants.%tuple.type.748, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   %.loc10_49.3: %facet_type = converted constants.%tuple.type.748, %facet_value.loc10_49.2 [concrete = constants.%facet_value.e9d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_49.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.ec5 = name_ref has_value, imports.%Core.import_ref.b24 [concrete = imports.%field_decl.6a4]
// CHECK:STDOUT:   %.loc10_49.4: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc10_49.3: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_49.5: %facet_type = converted bool, %facet_value.loc10_49.3 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %facet_value.loc10_49.4: %facet_type = facet_value bool, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   %.loc10_49.6: %facet_type = converted bool, %facet_value.loc10_49.4 [concrete = constants.%facet_value.1f4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_49.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_48(%self.param: ref %EmptyRange.f31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_48(%self.param: ref %EmptyRange.f31) = "no_op";
// CHECK:STDOUT:
