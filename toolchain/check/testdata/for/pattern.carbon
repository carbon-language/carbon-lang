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
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.898: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.b5a: %EmptyRange.Make.type.898 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %EmptyRange.2c8: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.a58: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.32b: %EmptyRange.Make.type.a58 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a31: type = pattern_type %EmptyRange.2c8 [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.32b, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.7ba: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.152: type = unbound_element_type %Optional.7ba, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.03d: type = unbound_element_type %Optional.7ba, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.16c: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.505: %Optional.Get.type.16c = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.39b: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c75: %Optional.HasValue.type.39b = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.d7b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.54c: %EmptyRange.as.Iterate.impl.Next.type.d7b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.e80: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.d13: %EmptyRange.as.Iterate.impl.NewCursor.type.e80 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.e4f: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.084: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.1ce: %EmptyRange.as.Iterate.impl.NewCursor.type.084 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.b25: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cda: %EmptyRange.as.Iterate.impl.Next.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.2c8, (%Iterate.impl_witness.e4f) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.229: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.530: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.a07: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.229, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.1ce, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.c5d: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.530, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.309: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.2e8: type = pattern_type %Optional.309 [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.cda, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.6ab: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.746: %Optional.HasValue.type.6ab = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f20: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.c7c: %Optional.Get.type.f20 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.5e9: type = unbound_element_type %Optional.309, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.f70: type = unbound_element_type %Optional.309, %C [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9e7: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.746, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.c7c, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_36.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc10_36.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_36.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_36.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc10_36.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_36.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_36.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc10_36.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_36.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1a1: type = pattern_type %struct_type.has_value.value.9e7 [concrete]
// CHECK:STDOUT:   %self.param_patt.c09: %pattern_type.1a1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.343: %pattern_type.1a1 = wrapper_binding_pattern self, %self.param_patt.c09 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_36.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc10_36.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.22e: %pattern_type.2e8 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.437: %pattern_type.2e8 = wrapper_binding_pattern self, %self.param_patt.22e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.7d8: %pattern_type.a31 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bdd: %pattern_type.a31 = wrapper_binding_pattern self, %self.param_patt.7d8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_35 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc10_35 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_35 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.6e9: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.898) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.b5a)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e7f: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.39b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c75)]
// CHECK:STDOUT:   %Core.import_ref.22a1: @Optional.%Optional.Get.type (%Optional.Get.type.16c) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.505)]
// CHECK:STDOUT:   %Core.import_ref.3702: @Optional.%Optional.elem.1 (%Optional.elem.03d) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.564]
// CHECK:STDOUT:   %Core.import_ref.947: @Optional.%Optional.elem.2 (%Optional.elem.152) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.827]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.a41: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.e80) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.d13)]
// CHECK:STDOUT:   %Main.import_ref.406: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.d7b) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.54c)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.a41, %Main.import_ref.406), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.827: @Optional.%Optional.elem.2 (%Optional.elem.152) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.564: @Optional.%Optional.elem.1 (%Optional.elem.03d) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_36.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc10_36.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_36.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc10_36.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_36.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc10_36.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_36.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a1 = ref_param_pattern [concrete = constants.%self.param_patt.c09]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.343]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.has_value.value.9e7 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.has_value.value.9e7 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc10_36.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a1 = ref_param_pattern [concrete = constants.%self.param_patt.c09]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.343]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.has_value.value.9e7 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.has_value.value.9e7 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_36.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2e8 = ref_param_pattern [concrete = constants.%self.param_patt.22e]
// CHECK:STDOUT:     %self.patt: %pattern_type.2e8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.437]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.309 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.309 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc10_36.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2e8 = ref_param_pattern [concrete = constants.%self.param_patt.22e]
// CHECK:STDOUT:     %self.patt: %pattern_type.2e8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.437]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.309 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.309 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_35 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a31 = ref_param_pattern [concrete = constants.%self.param_patt.7d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.a31 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bdd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.2c8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.2c8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc10_35 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a31 = ref_param_pattern [concrete = constants.%self.param_patt.7d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.a31 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bdd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.2c8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.2c8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.2c8]
// CHECK:STDOUT:   %.loc10_29: %EmptyRange.Make.type.a58 = specific_constant imports.%Main.import_ref.6e9, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.32b]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.a58 = name_ref Make, %.loc10_29 [concrete = constants.%EmptyRange.Make.32b]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_35.1: ref %EmptyRange.2c8 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.2c8 to %.loc10_35.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_35.2: ref %EmptyRange.2c8 = temporary %.loc10_35.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.a07 = impl_witness_access constants.%Iterate.impl_witness.e4f, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.1ce]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %.loc10_35.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_36.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.1
// CHECK:STDOUT:   %.loc10_35.3: %EmptyRange.2c8 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_36.2(%.loc10_35.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.c5d = impl_witness_access constants.%Iterate.impl_witness.e4f, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.cda]
// CHECK:STDOUT:   %bound_method.loc10_36.3: <bound method> = bound_method %.loc10_35.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_36.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.4: <bound method> = bound_method %.loc10_35.2, %specific_fn.loc10_36.2
// CHECK:STDOUT:   %.loc10_36.1: ref %Optional.309 = temporary_storage
// CHECK:STDOUT:   %.loc10_35.4: %EmptyRange.2c8 = acquire_value %.loc10_35.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.309 to %.loc10_36.1 = call %bound_method.loc10_36.4(%.loc10_35.4, %var)
// CHECK:STDOUT:   %.loc10_36.2: ref %Optional.309 = temporary %.loc10_36.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_36.3: %Optional.HasValue.type.6ab = specific_constant imports.%Core.import_ref.e7f, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.746]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.6ab = name_ref HasValue, %.loc10_36.3 [concrete = constants.%Optional.HasValue.746]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_36.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.5: <bound method> = bound_method %.loc10_36.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_36.4: %Optional.309 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_36.5(%.loc10_36.4)
// CHECK:STDOUT:   %.loc10_36.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_36.6: bool = converted %Optional.HasValue.call, %.loc10_36.5
// CHECK:STDOUT:   if %.loc10_36.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_36.7: %Optional.Get.type.f20 = specific_constant imports.%Core.import_ref.22a1, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.c7c]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f20 = name_ref Get, %.loc10_36.7 [concrete = constants.%Optional.Get.c7c]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_36.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.6: <bound method> = bound_method %.loc10_36.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_36.8: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_36.9: %Optional.309 = acquire_value %.loc10_36.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_36.8 = call %bound_method.loc10_36.6(%.loc10_36.9)
// CHECK:STDOUT:   %.loc10_36.10: ref %C = temporary %.loc10_36.8, %Optional.Get.call
// CHECK:STDOUT:   %.loc10_36.11: %C = acquire_value %.loc10_36.10
// CHECK:STDOUT:   %C.ref.loc10_11: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc10_36.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%c.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.10, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.1(%.loc10_36.10)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.2(%.loc10_36.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.3: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.3(%.loc10_36.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.4(%var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_35: <bound method> = bound_method %.loc10_35.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_35: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_35(%.loc10_35.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_36.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_36.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_36.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_36.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_36.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_36.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_36.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_36.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_36.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_36.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_36.1: ref %C = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.1(%.loc10_36.1)
// CHECK:STDOUT:   %.loc10_36.2: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.2(%.loc10_36.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_36.4(%self.param: ref %struct_type.has_value.value.9e7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_36.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_36.5(%self.param: ref %Optional.309) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.f70 = name_ref value, imports.%Core.import_ref.947 [concrete = imports.%field_decl.827]
// CHECK:STDOUT:   %.loc10_36.1: ref %C = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.1(%.loc10_36.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.5e9 = name_ref has_value, imports.%Core.import_ref.3702 [concrete = imports.%field_decl.564]
// CHECK:STDOUT:   %.loc10_36.2: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.2(%.loc10_36.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_36.5(%self.param: ref %Optional.309) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_36.5(%self.param: ref %Optional.309) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_35(%self.param: ref %EmptyRange.2c8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_35(%self.param: ref %EmptyRange.2c8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_35(%self.param: ref %EmptyRange.2c8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.898: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.b5a: %EmptyRange.Make.type.898 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.518: <witness> = impl_witness imports.%Copy.impl_witness_table.b4f [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %C, (%Copy.impl_witness.518) [concrete]
// CHECK:STDOUT:   %EmptyRange.2c8: type = class_type @EmptyRange, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.a58: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.32b: %EmptyRange.Make.type.a58 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a31: type = pattern_type %EmptyRange.2c8 [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.32b, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.7ba: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.152: type = unbound_element_type %Optional.7ba, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.03d: type = unbound_element_type %Optional.7ba, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.16c: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.505: %Optional.Get.type.16c = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.39b: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c75: %Optional.HasValue.type.39b = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.d7b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.54c: %EmptyRange.as.Iterate.impl.Next.type.d7b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.e80: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.d13: %EmptyRange.as.Iterate.impl.NewCursor.type.e80 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.e4f: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.084: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.1ce: %EmptyRange.as.Iterate.impl.NewCursor.type.084 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.b25: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.cda: %EmptyRange.as.Iterate.impl.Next.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.2c8, (%Iterate.impl_witness.e4f) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.229: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.530: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.a07: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.229, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.1ce, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.c5d: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.530, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.309: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.2e8: type = pattern_type %Optional.309 [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.cda, @EmptyRange.as.Iterate.impl.Next(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.6ab: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.746: %Optional.HasValue.type.6ab = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f20: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.c7c: %Optional.Get.type.f20 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.5e9: type = unbound_element_type %Optional.309, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.f70: type = unbound_element_type %Optional.309, %C [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9e7: type = struct_type {.has_value: bool, .value: %C} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.746, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.c7c, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_40.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc10_40.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_40.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_40.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc10_40.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_40.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_40.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc10_40.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_40.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1a1: type = pattern_type %struct_type.has_value.value.9e7 [concrete]
// CHECK:STDOUT:   %self.param_patt.c09: %pattern_type.1a1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.343: %pattern_type.1a1 = wrapper_binding_pattern self, %self.param_patt.c09 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_40.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc10_40.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.22e: %pattern_type.2e8 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.437: %pattern_type.2e8 = wrapper_binding_pattern self, %self.param_patt.22e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_40.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc10_40.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_40.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.7d8: %pattern_type.a31 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bdd: %pattern_type.a31 = wrapper_binding_pattern self, %self.param_patt.7d8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_39 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc10_39 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_39 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.6e9: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.898) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.b5a)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e7f: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.39b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c75)]
// CHECK:STDOUT:   %Core.import_ref.22a1: @Optional.%Optional.Get.type (%Optional.Get.type.16c) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.505)]
// CHECK:STDOUT:   %Core.import_ref.3702: @Optional.%Optional.elem.1 (%Optional.elem.03d) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.564]
// CHECK:STDOUT:   %Core.import_ref.947: @Optional.%Optional.elem.2 (%Optional.elem.152) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.827]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.a41: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.e80) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.d13)]
// CHECK:STDOUT:   %Main.import_ref.406: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.d7b) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.54c)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.a41, %Main.import_ref.406), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.827: @Optional.%Optional.elem.2 (%Optional.elem.152) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.564: @Optional.%Optional.elem.1 (%Optional.elem.03d) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_40.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc10_40.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_40.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc10_40.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_40.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc10_40.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_40.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a1 = ref_param_pattern [concrete = constants.%self.param_patt.c09]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.343]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.has_value.value.9e7 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.has_value.value.9e7 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc10_40.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a1 = ref_param_pattern [concrete = constants.%self.param_patt.c09]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.343]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.has_value.value.9e7 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.has_value.value.9e7 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_40.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2e8 = ref_param_pattern [concrete = constants.%self.param_patt.22e]
// CHECK:STDOUT:     %self.patt: %pattern_type.2e8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.437]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.309 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.309 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc10_40.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2e8 = ref_param_pattern [concrete = constants.%self.param_patt.22e]
// CHECK:STDOUT:     %self.patt: %pattern_type.2e8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.437]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.309 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.309 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_39 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a31 = ref_param_pattern [concrete = constants.%self.param_patt.7d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.a31 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bdd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.2c8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.2c8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc10_39 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a31 = ref_param_pattern [concrete = constants.%self.param_patt.7d8]
// CHECK:STDOUT:     %self.patt: %pattern_type.a31 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bdd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.2c8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.2c8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.2c8]
// CHECK:STDOUT:   %.loc10_33: %EmptyRange.Make.type.a58 = specific_constant imports.%Main.import_ref.6e9, @EmptyRange(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.32b]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.a58 = name_ref Make, %.loc10_33 [concrete = constants.%EmptyRange.Make.32b]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_39.1: ref %EmptyRange.2c8 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.2c8 to %.loc10_39.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_39.2: ref %EmptyRange.2c8 = temporary %.loc10_39.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.a07 = impl_witness_access constants.%Iterate.impl_witness.e4f, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.1ce]
// CHECK:STDOUT:   %bound_method.loc10_40.1: <bound method> = bound_method %.loc10_39.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_40.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.2: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.1
// CHECK:STDOUT:   %.loc10_39.3: %EmptyRange.2c8 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_40.2(%.loc10_39.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.c5d = impl_witness_access constants.%Iterate.impl_witness.e4f, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.cda]
// CHECK:STDOUT:   %bound_method.loc10_40.3: <bound method> = bound_method %.loc10_39.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_40.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.4: <bound method> = bound_method %.loc10_39.2, %specific_fn.loc10_40.2
// CHECK:STDOUT:   %.loc10_40.1: ref %Optional.309 = temporary_storage
// CHECK:STDOUT:   %.loc10_39.4: %EmptyRange.2c8 = acquire_value %.loc10_39.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.309 to %.loc10_40.1 = call %bound_method.loc10_40.4(%.loc10_39.4, %var)
// CHECK:STDOUT:   %.loc10_40.2: ref %Optional.309 = temporary %.loc10_40.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_40.3: %Optional.HasValue.type.6ab = specific_constant imports.%Core.import_ref.e7f, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.746]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.6ab = name_ref HasValue, %.loc10_40.3 [concrete = constants.%Optional.HasValue.746]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_40.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.5: <bound method> = bound_method %.loc10_40.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_40.4: %Optional.309 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_40.5(%.loc10_40.4)
// CHECK:STDOUT:   %.loc10_40.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_40.6: bool = converted %Optional.HasValue.call, %.loc10_40.5
// CHECK:STDOUT:   if %.loc10_40.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_40.7: %Optional.Get.type.f20 = specific_constant imports.%Core.import_ref.22a1, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.c7c]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f20 = name_ref Get, %.loc10_40.7 [concrete = constants.%Optional.Get.c7c]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_40.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_40.6: <bound method> = bound_method %.loc10_40.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.309 = acquire_value %.loc10_40.2
// CHECK:STDOUT:   %Optional.Get.call: init %C to %.loc10_8 = call %bound_method.loc10_40.6(%.loc10_40.8)
// CHECK:STDOUT:   assign %c.var, %Optional.Get.call
// CHECK:STDOUT:   %C.ref.loc10_15: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %addr: %ptr.6b6 = addr_of %c.ref
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.1(%.loc10_40.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_8.1: <bound method> = bound_method %c.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_8.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_8.1(%c.var)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.2: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.2(%.loc10_40.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_8.2: <bound method> = bound_method %c.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_8.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_8.2(%c.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.3: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.3(%var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_39: <bound method> = bound_method %.loc10_39.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_39: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_39(%.loc10_39.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_40.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_40.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_40.1(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_40.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_40.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_40.2(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_40.3(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_40.3(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_40.3(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_40.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_40.1: ref %C = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.1(%.loc10_40.1)
// CHECK:STDOUT:   %.loc10_40.2: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.2: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.2(%.loc10_40.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_40.4(%self.param: ref %struct_type.has_value.value.9e7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_40.4(%self.param: ref %struct_type.has_value.value.9e7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_40.5(%self.param: ref %Optional.309) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.f70 = name_ref value, imports.%Core.import_ref.947 [concrete = imports.%field_decl.827]
// CHECK:STDOUT:   %.loc10_40.1: ref %C = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.1(%.loc10_40.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.5e9 = name_ref has_value, imports.%Core.import_ref.3702 [concrete = imports.%field_decl.564]
// CHECK:STDOUT:   %.loc10_40.2: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.2: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.2(%.loc10_40.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_40.5(%self.param: ref %Optional.309) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_40.5(%self.param: ref %Optional.309) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_39(%self.param: ref %EmptyRange.2c8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_39(%self.param: ref %EmptyRange.2c8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_39(%self.param: ref %EmptyRange.2c8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.898: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.b5a: %EmptyRange.Make.type.898 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %EmptyRange.475: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.1bd: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.728: %EmptyRange.Make.type.1bd = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9fc: type = pattern_type %EmptyRange.475 [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.728, @EmptyRange.Make(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.7ba: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.152: type = unbound_element_type %Optional.7ba, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.03d: type = unbound_element_type %Optional.7ba, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.16c: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.505: %Optional.Get.type.16c = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.39b: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c75: %Optional.HasValue.type.39b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.d7b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.54c: %EmptyRange.as.Iterate.impl.Next.type.d7b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.e80: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.d13: %EmptyRange.as.Iterate.impl.NewCursor.type.e80 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.401: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.15d: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.51c: %EmptyRange.as.Iterate.impl.NewCursor.type.15d = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.e48: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.953: %EmptyRange.as.Iterate.impl.Next.type.e48 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.475, (%Iterate.impl_witness.401) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.2bb: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.461: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.0a9: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.2bb, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.51c, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %.e0f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.461, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.d47: type = class_type @Optional, @Optional(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %pattern_type.29d: type = pattern_type %Optional.d47 [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.953, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.5f7: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ef8: %Optional.HasValue.type.5f7 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.72e: type = fn_type @Optional.Get, @Optional(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Optional.Get.77b: %Optional.Get.type.72e = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.4e1: type = unbound_element_type %Optional.d47, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.b7c: type = unbound_element_type %Optional.d47, %tuple.type.784 [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.6f9: type = struct_type {.has_value: bool, .value: %tuple.type.784} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.ef8, @Optional.HasValue(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.77b, @Optional.Get(%Copy.facet.fa2) [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_61.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc10_61.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_61.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.fa3: %pattern_type.860 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.3e4: %pattern_type.860 = wrapper_binding_pattern self, %self.param_patt.fa3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_61.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc10_61.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_61.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.095: type = pattern_type %struct_type.has_value.value.6f9 [concrete]
// CHECK:STDOUT:   %self.param_patt.e70: %pattern_type.095 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cce: %pattern_type.095 = wrapper_binding_pattern self, %self.param_patt.e70 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_61.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc10_61.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.d3c: %pattern_type.29d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1fe: %pattern_type.29d = wrapper_binding_pattern self, %self.param_patt.d3c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_61.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc10_61.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_61.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_61.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc10_61.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_61.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.75c: %pattern_type.9fc = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b20: %pattern_type.9fc = wrapper_binding_pattern self, %self.param_patt.75c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_60 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc10_60 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_60 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.import_ref.6e9: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.898) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.b5a)]
// CHECK:STDOUT:   %Core.import_ref.08a: @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.74a) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.4af)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5a = impl_witness_table (%Core.import_ref.08a), @tuple.type.as.Copy.impl.9bd [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e7f: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.39b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c75)]
// CHECK:STDOUT:   %Core.import_ref.22a1: @Optional.%Optional.Get.type (%Optional.Get.type.16c) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.505)]
// CHECK:STDOUT:   %Core.import_ref.3702: @Optional.%Optional.elem.1 (%Optional.elem.03d) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.564]
// CHECK:STDOUT:   %Core.import_ref.947: @Optional.%Optional.elem.2 (%Optional.elem.152) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.827]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.a41: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.e80) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.d13)]
// CHECK:STDOUT:   %Main.import_ref.406: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.d7b) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.54c)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.a41, %Main.import_ref.406), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.827: @Optional.%Optional.elem.2 (%Optional.elem.152) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.564: @Optional.%Optional.elem.1 (%Optional.elem.03d) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_61.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc10_61.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_61.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.860 = ref_param_pattern [concrete = constants.%self.param_patt.fa3]
// CHECK:STDOUT:     %self.patt: %pattern_type.860 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3e4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.784 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.784 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc10_61.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.860 = ref_param_pattern [concrete = constants.%self.param_patt.fa3]
// CHECK:STDOUT:     %self.patt: %pattern_type.860 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3e4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.784 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.784 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_61.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.095 = ref_param_pattern [concrete = constants.%self.param_patt.e70]
// CHECK:STDOUT:     %self.patt: %pattern_type.095 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.has_value.value.6f9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.has_value.value.6f9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc10_61.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.095 = ref_param_pattern [concrete = constants.%self.param_patt.e70]
// CHECK:STDOUT:     %self.patt: %pattern_type.095 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.has_value.value.6f9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.has_value.value.6f9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_61.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.29d = ref_param_pattern [concrete = constants.%self.param_patt.d3c]
// CHECK:STDOUT:     %self.patt: %pattern_type.29d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1fe]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.d47 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.d47 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc10_61.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.29d = ref_param_pattern [concrete = constants.%self.param_patt.d3c]
// CHECK:STDOUT:     %self.patt: %pattern_type.29d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1fe]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.d47 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.d47 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_61.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc10_61.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_60 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9fc = ref_param_pattern [concrete = constants.%self.param_patt.75c]
// CHECK:STDOUT:     %self.patt: %pattern_type.9fc = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b20]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.475 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.475 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc10_60 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9fc = ref_param_pattern [concrete = constants.%self.param_patt.75c]
// CHECK:STDOUT:     %self.patt: %pattern_type.9fc = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b20]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.475 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.475 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.475]
// CHECK:STDOUT:   %.loc10_54: %EmptyRange.Make.type.1bd = specific_constant imports.%Main.import_ref.6e9, @EmptyRange(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.Make.728]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.1bd = name_ref Make, %.loc10_54 [concrete = constants.%EmptyRange.Make.728]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_60.1: ref %EmptyRange.475 = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.475 to %.loc10_60.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_60.2: ref %EmptyRange.475 = temporary %.loc10_60.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.0a9 = impl_witness_access constants.%Iterate.impl_witness.401, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.51c]
// CHECK:STDOUT:   %bound_method.loc10_61.1: <bound method> = bound_method %.loc10_60.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_61.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.2: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.1
// CHECK:STDOUT:   %.loc10_60.3: %EmptyRange.475 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_61.2(%.loc10_60.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.e0f = impl_witness_access constants.%Iterate.impl_witness.401, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.953]
// CHECK:STDOUT:   %bound_method.loc10_61.3: <bound method> = bound_method %.loc10_60.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_61.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.fa2) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.4: <bound method> = bound_method %.loc10_60.2, %specific_fn.loc10_61.2
// CHECK:STDOUT:   %.loc10_61.1: ref %Optional.d47 = temporary_storage
// CHECK:STDOUT:   %.loc10_60.4: %EmptyRange.475 = acquire_value %.loc10_60.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.d47 to %.loc10_61.1 = call %bound_method.loc10_61.4(%.loc10_60.4, %var)
// CHECK:STDOUT:   %.loc10_61.2: ref %Optional.d47 = temporary %.loc10_61.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_61.3: %Optional.HasValue.type.5f7 = specific_constant imports.%Core.import_ref.e7f, @Optional(constants.%Copy.facet.fa2) [concrete = constants.%Optional.HasValue.ef8]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.5f7 = name_ref HasValue, %.loc10_61.3 [concrete = constants.%Optional.HasValue.ef8]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_61.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.fa2) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.5: <bound method> = bound_method %.loc10_61.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_61.4: %Optional.d47 = acquire_value %.loc10_61.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_61.5(%.loc10_61.4)
// CHECK:STDOUT:   %.loc10_61.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_61.6: bool = converted %Optional.HasValue.call, %.loc10_61.5
// CHECK:STDOUT:   if %.loc10_61.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_61.7: %Optional.Get.type.72e = specific_constant imports.%Core.import_ref.22a1, @Optional(constants.%Copy.facet.fa2) [concrete = constants.%Optional.Get.77b]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.72e = name_ref Get, %.loc10_61.7 [concrete = constants.%Optional.Get.77b]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_61.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.fa2) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_61.6: <bound method> = bound_method %.loc10_61.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_61.8: ref %tuple.type.784 = temporary_storage
// CHECK:STDOUT:   %.loc10_61.9: %Optional.d47 = acquire_value %.loc10_61.2
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.10, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1(%.loc10_61.10)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2(%.loc10_61.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.3: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.3(%.loc10_61.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.4(%var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_60: <bound method> = bound_method %.loc10_60.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_60: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_60(%.loc10_60.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_61.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_61.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_61.1(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_61.2(%self.param: ref %tuple.type.784) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref bool = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_61.2(%self.param: ref %tuple.type.784) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_61.2(%self.param: ref %tuple.type.784) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_61.3(%self.param: ref %struct_type.has_value.value.6f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_61.1: ref %tuple.type.784 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1(%.loc10_61.1)
// CHECK:STDOUT:   %.loc10_61.2: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2(%.loc10_61.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_61.3(%self.param: ref %struct_type.has_value.value.6f9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_61.3(%self.param: ref %struct_type.has_value.value.6f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_61.4(%self.param: ref %Optional.d47) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.b7c = name_ref value, imports.%Core.import_ref.947 [concrete = imports.%field_decl.827]
// CHECK:STDOUT:   %.loc10_61.1: ref %tuple.type.784 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1(%.loc10_61.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.4e1 = name_ref has_value, imports.%Core.import_ref.3702 [concrete = imports.%field_decl.564]
// CHECK:STDOUT:   %.loc10_61.2: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2(%.loc10_61.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_61.4(%self.param: ref %Optional.d47) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_61.4(%self.param: ref %Optional.d47) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_61.5(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_61.5(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_61.5(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_60(%self.param: ref %EmptyRange.475) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_60(%self.param: ref %EmptyRange.475) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_60(%self.param: ref %EmptyRange.475) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %EmptyRange.type: type = generic_class_type @EmptyRange [concrete]
// CHECK:STDOUT:   %EmptyRange.generic: %EmptyRange.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.type.898: type = fn_type @EmptyRange.Make, @EmptyRange(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.Make.b5a: %EmptyRange.Make.type.898 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %EmptyRange.61c: type = class_type @EmptyRange, @EmptyRange(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.type.9f5: type = fn_type @EmptyRange.Make, @EmptyRange(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.98c: %EmptyRange.Make.type.9f5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.df0: type = pattern_type %EmptyRange.61c [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.98c, @EmptyRange.Make(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.7ba: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.152: type = unbound_element_type %Optional.7ba, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.03d: type = unbound_element_type %Optional.7ba, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.16c: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.505: %Optional.Get.type.16c = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.39b: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c75: %Optional.HasValue.type.39b = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.d7b: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.54c: %EmptyRange.as.Iterate.impl.Next.type.d7b = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.e80: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.d13: %EmptyRange.as.Iterate.impl.NewCursor.type.e80 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.e2e: <witness> = impl_witness imports.%Iterate.impl_witness_table, @EmptyRange.as.Iterate.impl(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.f98: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.2d7: %EmptyRange.as.Iterate.impl.NewCursor.type.f98 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.43f: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.c89: %EmptyRange.as.Iterate.impl.Next.type.43f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %EmptyRange.61c, (%Iterate.impl_witness.e2e) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.07d: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.871: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [concrete]
// CHECK:STDOUT:   %.8c3: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.07d, %Iterate.facet [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.NewCursor.2d7, @EmptyRange.as.Iterate.impl.NewCursor(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %.db4: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.871, %Iterate.facet [concrete]
// CHECK:STDOUT:   %Optional.c97: type = class_type @Optional, @Optional(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %pattern_type.5df: type = pattern_type %Optional.c97 [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %EmptyRange.as.Iterate.impl.Next.c89, @EmptyRange.as.Iterate.impl.Next(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.181: type = fn_type @Optional.HasValue, @Optional(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.2c0: %Optional.HasValue.type.181 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.50f: type = fn_type @Optional.Get, @Optional(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Optional.Get.2f5: %Optional.Get.type.50f = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.elem.4bd: type = unbound_element_type %Optional.c97, bool [concrete]
// CHECK:STDOUT:   %Optional.elem.720: type = unbound_element_type %Optional.c97, %tuple.type.748 [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.9b1: type = struct_type {.has_value: bool, .value: %tuple.type.748} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.2c0, @Optional.HasValue(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.2f5, @Optional.Get(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_49.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc10_49.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_49.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_49.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc10_49.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_49.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.a49: %pattern_type.8b4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d80: %pattern_type.8b4 = wrapper_binding_pattern self, %self.param_patt.a49 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_49.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc10_49.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_49.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_49.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc10_49.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_49.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ebb: type = pattern_type %struct_type.has_value.value.9b1 [concrete]
// CHECK:STDOUT:   %self.param_patt.c1b: %pattern_type.ebb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4fb: %pattern_type.ebb = wrapper_binding_pattern self, %self.param_patt.c1b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_49.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc10_49.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.6cf: %pattern_type.5df = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.697: %pattern_type.5df = wrapper_binding_pattern self, %self.param_patt.6cf [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.c7a: %pattern_type.df0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a77: %pattern_type.df0 = wrapper_binding_pattern self, %self.param_patt.c7a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_48 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc10_48 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.7: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_48 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.7: %Destroy.WithSelf.SelfDestruct.type.fbceb5.7 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: %C.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.EmptyRange: %EmptyRange.type = import_ref Main//empty_range, EmptyRange, loaded [concrete = constants.%EmptyRange.generic]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//empty_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.6e9: @EmptyRange.%EmptyRange.Make.type (%EmptyRange.Make.type.898) = import_ref Main//empty_range, loc5_21, loaded [symbolic = @EmptyRange.%EmptyRange.Make (constants.%EmptyRange.Make.b5a)]
// CHECK:STDOUT:   %Core.import_ref.08a: @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op.type (%tuple.type.as.Copy.impl.Op.type.74a) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @tuple.type.as.Copy.impl.9bd.%tuple.type.as.Copy.impl.Op (constants.%tuple.type.as.Copy.impl.Op.4af)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5a = impl_witness_table (%Core.import_ref.08a), @tuple.type.as.Copy.impl.9bd [concrete]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.Copy.impl.Op.type = import_ref Main//empty_range, loc19_32, loaded [concrete = constants.%C.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b4f = impl_witness_table (%Main.import_ref.341), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e7f: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.39b) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c75)]
// CHECK:STDOUT:   %Core.import_ref.22a1: @Optional.%Optional.Get.type (%Optional.Get.type.16c) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.505)]
// CHECK:STDOUT:   %Core.import_ref.3702: @Optional.%Optional.elem.1 (%Optional.elem.03d) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.564]
// CHECK:STDOUT:   %Core.import_ref.947: @Optional.%Optional.elem.2 (%Optional.elem.152) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.827]
// CHECK:STDOUT:   %Main.import_ref.71a = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.999 = import_ref Main//empty_range, loc7_68, unloaded
// CHECK:STDOUT:   %Main.import_ref.a41: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor.type (%EmptyRange.as.Iterate.impl.NewCursor.type.e80) = import_ref Main//empty_range, loc8_37, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.NewCursor (constants.%EmptyRange.as.Iterate.impl.NewCursor.d13)]
// CHECK:STDOUT:   %Main.import_ref.406: @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next.type (%EmptyRange.as.Iterate.impl.Next.type.d7b) = import_ref Main//empty_range, loc11_69, loaded [symbolic = @EmptyRange.as.Iterate.impl.%EmptyRange.as.Iterate.impl.Next (constants.%EmptyRange.as.Iterate.impl.Next.54c)]
// CHECK:STDOUT:   %Iterate.impl_witness_table = impl_witness_table (%Main.import_ref.71a, %Main.import_ref.999, %Main.import_ref.a41, %Main.import_ref.406), @EmptyRange.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.827: @Optional.%Optional.elem.2 (%Optional.elem.152) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.564: @Optional.%Optional.elem.1 (%Optional.elem.03d) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_49.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc10_49.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_49.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc10_49.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_49.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8b4 = ref_param_pattern [concrete = constants.%self.param_patt.a49]
// CHECK:STDOUT:     %self.patt: %pattern_type.8b4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d80]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.748 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.748 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc10_49.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8b4 = ref_param_pattern [concrete = constants.%self.param_patt.a49]
// CHECK:STDOUT:     %self.patt: %pattern_type.8b4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d80]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.748 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.748 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_49.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc10_49.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_49.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ebb = ref_param_pattern [concrete = constants.%self.param_patt.c1b]
// CHECK:STDOUT:     %self.patt: %pattern_type.ebb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4fb]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.has_value.value.9b1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.has_value.value.9b1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc10_49.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ebb = ref_param_pattern [concrete = constants.%self.param_patt.c1b]
// CHECK:STDOUT:     %self.patt: %pattern_type.ebb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4fb]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.has_value.value.9b1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.has_value.value.9b1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_49.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5df = ref_param_pattern [concrete = constants.%self.param_patt.6cf]
// CHECK:STDOUT:     %self.patt: %pattern_type.5df = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.697]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.c97 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.c97 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc10_49.6 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5df = ref_param_pattern [concrete = constants.%self.param_patt.6cf]
// CHECK:STDOUT:     %self.patt: %pattern_type.5df = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.697]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.c97 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.c97 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_48 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df0 = ref_param_pattern [concrete = constants.%self.param_patt.c7a]
// CHECK:STDOUT:     %self.patt: %pattern_type.df0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a77]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.61c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.61c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc10_48 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df0 = ref_param_pattern [concrete = constants.%self.param_patt.c7a]
// CHECK:STDOUT:     %self.patt: %pattern_type.df0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a77]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.61c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.61c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %EmptyRange: type = class_type @EmptyRange, @EmptyRange(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.61c]
// CHECK:STDOUT:   %.loc10_42: %EmptyRange.Make.type.9f5 = specific_constant imports.%Main.import_ref.6e9, @EmptyRange(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.Make.98c]
// CHECK:STDOUT:   %Make.ref: %EmptyRange.Make.type.9f5 = name_ref Make, %.loc10_42 [concrete = constants.%EmptyRange.Make.98c]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %Make.ref, @EmptyRange.Make(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.Make.specific_fn]
// CHECK:STDOUT:   %.loc10_48.1: ref %EmptyRange.61c = temporary_storage
// CHECK:STDOUT:   %EmptyRange.Make.call: init %EmptyRange.61c to %.loc10_48.1 = call %EmptyRange.Make.specific_fn()
// CHECK:STDOUT:   %.loc10_48.2: ref %EmptyRange.61c = temporary %.loc10_48.1, %EmptyRange.Make.call
// CHECK:STDOUT:   %impl.elem2: %.8c3 = impl_witness_access constants.%Iterate.impl_witness.e2e, element2 [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.2d7]
// CHECK:STDOUT:   %bound_method.loc10_49.1: <bound method> = bound_method %.loc10_48.2, %impl.elem2
// CHECK:STDOUT:   %specific_fn.loc10_49.1: <specific function> = specific_function %impl.elem2, @EmptyRange.as.Iterate.impl.NewCursor(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.2: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.1
// CHECK:STDOUT:   %.loc10_48.3: %EmptyRange.61c = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.call: init %empty_struct_type = call %bound_method.loc10_49.2(%.loc10_48.3)
// CHECK:STDOUT:   %var: ref %empty_struct_type = var_storage invalid
// CHECK:STDOUT:   assign %var, %EmptyRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.db4 = impl_witness_access constants.%Iterate.impl_witness.e2e, element3 [concrete = constants.%EmptyRange.as.Iterate.impl.Next.c89]
// CHECK:STDOUT:   %bound_method.loc10_49.3: <bound method> = bound_method %.loc10_48.2, %impl.elem3
// CHECK:STDOUT:   %specific_fn.loc10_49.2: <specific function> = specific_function %impl.elem3, @EmptyRange.as.Iterate.impl.Next(constants.%Copy.facet.83b) [concrete = constants.%EmptyRange.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.4: <bound method> = bound_method %.loc10_48.2, %specific_fn.loc10_49.2
// CHECK:STDOUT:   %.loc10_49.1: ref %Optional.c97 = temporary_storage
// CHECK:STDOUT:   %.loc10_48.4: %EmptyRange.61c = acquire_value %.loc10_48.2
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.call: init %Optional.c97 to %.loc10_49.1 = call %bound_method.loc10_49.4(%.loc10_48.4, %var)
// CHECK:STDOUT:   %.loc10_49.2: ref %Optional.c97 = temporary %.loc10_49.1, %EmptyRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc10_49.3: %Optional.HasValue.type.181 = specific_constant imports.%Core.import_ref.e7f, @Optional(constants.%Copy.facet.83b) [concrete = constants.%Optional.HasValue.2c0]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.181 = name_ref HasValue, %.loc10_49.3 [concrete = constants.%Optional.HasValue.2c0]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc10_49.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet.83b) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.5: <bound method> = bound_method %.loc10_49.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc10_49.4: %Optional.c97 = acquire_value %.loc10_49.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc10_49.5(%.loc10_49.4)
// CHECK:STDOUT:   %.loc10_49.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc10_49.6: bool = converted %Optional.HasValue.call, %.loc10_49.5
// CHECK:STDOUT:   if %.loc10_49.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc10_49.7: %Optional.Get.type.50f = specific_constant imports.%Core.import_ref.22a1, @Optional(constants.%Copy.facet.83b) [concrete = constants.%Optional.Get.2f5]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.50f = name_ref Get, %.loc10_49.7 [concrete = constants.%Optional.Get.2f5]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc10_49.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet.83b) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_49.6: <bound method> = bound_method %.loc10_49.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc10_49.8: ref %tuple.type.748 = temporary_storage
// CHECK:STDOUT:   %.loc10_49.9: %Optional.c97 = acquire_value %.loc10_49.2
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.10, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1(%.loc10_49.10)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2(%.loc10_49.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.3: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.3(%.loc10_49.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.4(%var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_48: <bound method> = bound_method %.loc10_48.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_48: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_48(%.loc10_48.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_49.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_49.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_49.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_49.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_49.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_49.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_49.3(%self.param: ref %tuple.type.748) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %C = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %C = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_49.3(%self.param: ref %tuple.type.748) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_49.3(%self.param: ref %tuple.type.748) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_49.4(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_49.4(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_49.4(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_49.5(%self.param: ref %struct_type.has_value.value.9b1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_49.1: ref %tuple.type.748 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1(%.loc10_49.1)
// CHECK:STDOUT:   %.loc10_49.2: ref bool = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2(%.loc10_49.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_49.5(%self.param: ref %struct_type.has_value.value.9b1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_49.5(%self.param: ref %struct_type.has_value.value.9b1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_49.6(%self.param: ref %Optional.c97) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.720 = name_ref value, imports.%Core.import_ref.947 [concrete = imports.%field_decl.827]
// CHECK:STDOUT:   %.loc10_49.1: ref %tuple.type.748 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1(%.loc10_49.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.4bd = name_ref has_value, imports.%Core.import_ref.3702 [concrete = imports.%field_decl.564]
// CHECK:STDOUT:   %.loc10_49.2: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2(%.loc10_49.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_49.6(%self.param: ref %Optional.c97) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_49.6(%self.param: ref %Optional.c97) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_48(%self.param: ref %EmptyRange.61c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_48(%self.param: ref %EmptyRange.61c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_48(%self.param: ref %EmptyRange.61c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
