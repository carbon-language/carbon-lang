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
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+4]]:7-43: error: cannot implicitly convert expression of type `C` to `X` [ConversionFailure]
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+3]]:7-43: note: type `C` does not implement interface `Core.ImplicitAs(X)` [MissingImplForType]
  // CHECK:STDERR: fail_bad_pattern.carbon:[[@LINE+2]]:15-18: note: initializing this binding [InPatternInit]
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
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type %EmptyRange.c31 [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.82d, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.afc: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.a1f: type = unbound_element_type %Optional.afc, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.fbd: type = unbound_element_type %Optional.afc, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f7a: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.c2b: %Optional.Get.type.f7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.23a: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f8f: %Optional.HasValue.type.23a = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.e76: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.706: %EmptyRange.as.Iterate.impl.Next.type.e76 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c99: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.f76: %EmptyRange.as.Iterate.impl.NewCursor.type.c99 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
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
// CHECK:STDOUT:   %pattern_type.a80: type = pattern_type %Optional.5b4 [concrete]
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
// CHECK:STDOUT:   %self.param_patt.c14: %pattern_type.a80 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.000: %pattern_type.a80 = wrapper_binding_pattern self, %self.param_patt.c14 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_36.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b71: %pattern_type.dc0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c0c: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt.b71 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.b75: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.897: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a80 = ref_param_pattern [concrete = constants.%self.param_patt.c14]
// CHECK:STDOUT:     %self.patt: %pattern_type.a80 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.000]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.5b4 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.5b4 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc10_36.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a80 = ref_param_pattern [concrete = constants.%self.param_patt.c14]
// CHECK:STDOUT:     %self.patt: %pattern_type.a80 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.000]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.5b4 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.5b4 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_35 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = ref_param_pattern [concrete = constants.%self.param_patt.b71]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.c31 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.c31 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc10_35 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = ref_param_pattern [concrete = constants.%self.param_patt.b71]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.c31 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.c31 = wrapper_binding self, %self.param
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_36.5(%self.param: ref %Optional.5b4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.433 = name_ref value, imports.%Core.import_ref.2f6 [concrete = imports.%field_decl.b75]
// CHECK:STDOUT:   %.loc10_36.1: ref %C = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.1: <bound method> = bound_method %.loc10_36.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.1(%.loc10_36.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.606 = name_ref has_value, imports.%Core.import_ref.5d0 [concrete = imports.%field_decl.897]
// CHECK:STDOUT:   %.loc10_36.2: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_36.2: <bound method> = bound_method %.loc10_36.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_36.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_36.2(%.loc10_36.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_36.5(%self.param: ref %Optional.5b4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_36.5(%self.param: ref %Optional.5b4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_35(%self.param: ref %EmptyRange.c31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_35(%self.param: ref %EmptyRange.c31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_35(%self.param: ref %EmptyRange.c31) {
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
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type %EmptyRange.c31 [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.82d, @EmptyRange.Make(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.afc: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.a1f: type = unbound_element_type %Optional.afc, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.fbd: type = unbound_element_type %Optional.afc, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f7a: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.c2b: %Optional.Get.type.f7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.23a: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f8f: %Optional.HasValue.type.23a = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.e76: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.706: %EmptyRange.as.Iterate.impl.Next.type.e76 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c99: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.f76: %EmptyRange.as.Iterate.impl.NewCursor.type.c99 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
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
// CHECK:STDOUT:   %pattern_type.a80: type = pattern_type %Optional.5b4 [concrete]
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
// CHECK:STDOUT:   %self.param_patt.c14: %pattern_type.a80 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.000: %pattern_type.a80 = wrapper_binding_pattern self, %self.param_patt.c14 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_40.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc10_40.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_40.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b71: %pattern_type.dc0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c0c: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt.b71 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.b75: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.897: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a80 = ref_param_pattern [concrete = constants.%self.param_patt.c14]
// CHECK:STDOUT:     %self.patt: %pattern_type.a80 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.000]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.5b4 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.5b4 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc10_40.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a80 = ref_param_pattern [concrete = constants.%self.param_patt.c14]
// CHECK:STDOUT:     %self.patt: %pattern_type.a80 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.000]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.5b4 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.5b4 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_39 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = ref_param_pattern [concrete = constants.%self.param_patt.b71]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.c31 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.c31 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc10_39 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = ref_param_pattern [concrete = constants.%self.param_patt.b71]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.c31 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.c31 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   %.loc10_8: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %.loc10_40.8: %Optional.5b4 = acquire_value %.loc10_40.2
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_40.5(%self.param: ref %Optional.5b4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.433 = name_ref value, imports.%Core.import_ref.2f6 [concrete = imports.%field_decl.b75]
// CHECK:STDOUT:   %.loc10_40.1: ref %C = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.1: <bound method> = bound_method %.loc10_40.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.1(%.loc10_40.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.606 = name_ref has_value, imports.%Core.import_ref.5d0 [concrete = imports.%field_decl.897]
// CHECK:STDOUT:   %.loc10_40.2: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_40.2: <bound method> = bound_method %.loc10_40.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_40.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_40.2(%.loc10_40.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_40.5(%self.param: ref %Optional.5b4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_40.5(%self.param: ref %Optional.5b4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_39(%self.param: ref %EmptyRange.c31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_39(%self.param: ref %EmptyRange.c31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_39(%self.param: ref %EmptyRange.c31) {
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
// CHECK:STDOUT:   %pattern_type.0ff: type = pattern_type %EmptyRange.7d6 [concrete]
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
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
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
// CHECK:STDOUT:   %pattern_type.b8c: type = pattern_type %Optional.9ca [concrete]
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
// CHECK:STDOUT:   %self.param_patt.b36: %pattern_type.b8c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbe: %pattern_type.b8c = wrapper_binding_pattern self, %self.param_patt.b36 [concrete]
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
// CHECK:STDOUT:   %self.param_patt.34c: %pattern_type.0ff = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b63: %pattern_type.0ff = wrapper_binding_pattern self, %self.param_patt.34c [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.b75: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.897: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b8c = ref_param_pattern [concrete = constants.%self.param_patt.b36]
// CHECK:STDOUT:     %self.patt: %pattern_type.b8c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbe]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.9ca = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.9ca = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc10_61.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b8c = ref_param_pattern [concrete = constants.%self.param_patt.b36]
// CHECK:STDOUT:     %self.patt: %pattern_type.b8c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbe]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.9ca = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.9ca = wrapper_binding self, %self.param
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0ff = ref_param_pattern [concrete = constants.%self.param_patt.34c]
// CHECK:STDOUT:     %self.patt: %pattern_type.0ff = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b63]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.7d6 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.7d6 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc10_60 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0ff = ref_param_pattern [concrete = constants.%self.param_patt.34c]
// CHECK:STDOUT:     %self.patt: %pattern_type.0ff = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b63]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.7d6 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.7d6 = wrapper_binding self, %self.param
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_61.4(%self.param: ref %Optional.9ca) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.6d0 = name_ref value, imports.%Core.import_ref.2f6 [concrete = imports.%field_decl.b75]
// CHECK:STDOUT:   %.loc10_61.1: ref %tuple.type.784 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1: <bound method> = bound_method %.loc10_61.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.1(%.loc10_61.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.79c = name_ref has_value, imports.%Core.import_ref.5d0 [concrete = imports.%field_decl.897]
// CHECK:STDOUT:   %.loc10_61.2: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2: <bound method> = bound_method %.loc10_61.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_61.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_61.2(%.loc10_61.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_61.4(%self.param: ref %Optional.9ca) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_61.4(%self.param: ref %Optional.9ca) {
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_60(%self.param: ref %EmptyRange.7d6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_60(%self.param: ref %EmptyRange.7d6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_60(%self.param: ref %EmptyRange.7d6) {
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
// CHECK:STDOUT:   %pattern_type.242: type = pattern_type %EmptyRange.ad6 [concrete]
// CHECK:STDOUT:   %EmptyRange.Make.specific_fn: <specific function> = specific_function %EmptyRange.Make.d66, @EmptyRange.Make(%Copy.facet.83b) [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Optional.afc: type = class_type @Optional, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.elem.a1f: type = unbound_element_type %Optional.afc, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Optional.elem.fbd: type = unbound_element_type %Optional.afc, bool [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f7a: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.c2b: %Optional.Get.type.f7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.23a: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f8f: %Optional.HasValue.type.23a = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.type.e76: type = fn_type @EmptyRange.as.Iterate.impl.Next, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.Next.706: %EmptyRange.as.Iterate.impl.Next.type.e76 = struct_value () [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.type.c99: type = fn_type @EmptyRange.as.Iterate.impl.NewCursor, @EmptyRange.as.Iterate.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %EmptyRange.as.Iterate.impl.NewCursor.f76: %EmptyRange.as.Iterate.impl.NewCursor.type.c99 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
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
// CHECK:STDOUT:   %pattern_type.19f: type = pattern_type %Optional.cc1 [concrete]
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
// CHECK:STDOUT:   %self.param_patt.6d0: %pattern_type.19f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f8d: %pattern_type.19f = wrapper_binding_pattern self, %self.param_patt.6d0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10_49.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.f63: %pattern_type.242 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cd1: %pattern_type.242 = wrapper_binding_pattern self, %self.param_patt.f63 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.127 = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.b75: @Optional.%Optional.elem.2 (%Optional.elem.a1f) = field_decl value, element1, %Core.import_ref.127 in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/parts/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.897: @Optional.%Optional.elem.1 (%Optional.elem.fbd) = field_decl has_value, element0, %Core.import_ref.b0d in [concrete] {}
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.19f = ref_param_pattern [concrete = constants.%self.param_patt.6d0]
// CHECK:STDOUT:     %self.patt: %pattern_type.19f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f8d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.cc1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.cc1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc10_49.6 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.19f = ref_param_pattern [concrete = constants.%self.param_patt.6d0]
// CHECK:STDOUT:     %self.patt: %pattern_type.19f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f8d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.cc1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.cc1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10_48 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.242 = ref_param_pattern [concrete = constants.%self.param_patt.f63]
// CHECK:STDOUT:     %self.patt: %pattern_type.242 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cd1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.ad6 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.ad6 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc10_48 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.242 = ref_param_pattern [concrete = constants.%self.param_patt.f63]
// CHECK:STDOUT:     %self.patt: %pattern_type.242 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cd1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %EmptyRange.ad6 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %EmptyRange.ad6 = wrapper_binding self, %self.param
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_49.6(%self.param: ref %Optional.cc1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Optional.elem.260 = name_ref value, imports.%Core.import_ref.2f6 [concrete = imports.%field_decl.b75]
// CHECK:STDOUT:   %.loc10_49.1: ref %tuple.type.748 = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1: <bound method> = bound_method %.loc10_49.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.1(%.loc10_49.1)
// CHECK:STDOUT:   %has_value.ref: %Optional.elem.c4d = name_ref has_value, imports.%Core.import_ref.5d0 [concrete = imports.%field_decl.897]
// CHECK:STDOUT:   %.loc10_49.2: ref bool = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2: <bound method> = bound_method %.loc10_49.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_49.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc10_49.2(%.loc10_49.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_49.6(%self.param: ref %Optional.cc1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_49.6(%self.param: ref %Optional.cc1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10_48(%self.param: ref %EmptyRange.ad6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10_48(%self.param: ref %EmptyRange.ad6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10_48(%self.param: ref %EmptyRange.ad6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
