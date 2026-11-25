// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/for.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/for/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/for/basic.carbon

// --- fail_not_range.carbon

library "[[@TEST_NAME]]";

fn Run() {
  // TODO: These diagnostics could be better.
  // CHECK:STDERR:        ╭─┤ fail_not_range.carbon:[[@LINE+8]]:7
  // CHECK:STDERR: ->     │   for (unused c: {} in {}) {
  // CHECK:STDERR:        ·       ┳━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: cannot access member of interface `Core.Iterate` in type `{}`
  // CHECK:STDERR:        ·          ┃        that does not implement that interface
  // CHECK:STDERR:        ·          ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  for (unused c: {} in {}) {
  }
}

// --- trivial.carbon

library "[[@TEST_NAME]]";

class TrivialRange {
  impl as Core.Iterate where .CursorType = () and .ElementType = () {
    fn NewCursor(unused self) {}
    fn Next(unused self, unused ref cursor: ()) -> Core.Optional(()) {
      return Core.Optional(()).None();
    }
  }
}

fn Body();
fn AfterLoop();

fn Run() {
  //@dump-sem-ir-begin
  for (_: () in {} as TrivialRange) {
    Body();
  }
  AfterLoop();
  //@dump-sem-ir-end
}


// CHECK:STDOUT: --- trivial.carbon = $trivial
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialRange: type = class_type @TrivialRange [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.f7a: type = fn_type @Optional.Get, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.Get.c2b: %Optional.Get.type.f7a = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.23a: type = fn_type @Optional.HasValue, @Optional(%T.f84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f8f: %Optional.HasValue.type.23a = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.37a: <witness> = impl_witness imports.%Copy.impl_witness_table.229 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %empty_tuple.type, (%Copy.impl_witness.37a) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness: <witness> = impl_witness @TrivialRange.as.Iterate.impl.%Iterate.impl_witness_table [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.type.de32c1.1: type = fn_type @TrivialRange.as.Iterate.impl.NewCursor.loc6_31.1 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.93de00.1: %TrivialRange.as.Iterate.impl.NewCursor.type.de32c1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Optional.9a5: type = class_type @Optional, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.type: type = fn_type @TrivialRange.as.Iterate.impl.Next [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next: %TrivialRange.as.Iterate.impl.Next.type = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.38e: %Iterate.type = facet_value %TrivialRange, (%Iterate.impl_witness) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.937: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.38e) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.741: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.38e) [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.type.de32c1.2: type = fn_type @TrivialRange.as.Iterate.impl.NewCursor.loc6_31.2 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.93de00.2: %TrivialRange.as.Iterate.impl.NewCursor.type.de32c1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.type.871: type = fn_type @Optional.HasValue, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.e1d: %Optional.HasValue.type.871 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.76d: type = fn_type @Optional.Get, @Optional(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.fd5: %Optional.Get.type.76d = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.has_value.value.da2: type = struct_type {.has_value: bool, .value: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Body.type: type = fn_type @Body [concrete]
// CHECK:STDOUT:   %Body: %Body.type = struct_value () [concrete]
// CHECK:STDOUT:   %AfterLoop.type: type = fn_type @AfterLoop [concrete]
// CHECK:STDOUT:   %AfterLoop: %AfterLoop.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.cb1 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialRange.val: %TrivialRange = struct_value () [concrete]
// CHECK:STDOUT:   %.50e: ref %TrivialRange = temporary invalid, %TrivialRange.val [concrete]
// CHECK:STDOUT:   %.953: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.937, %Iterate.facet.38e [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bound.464: <bound method> = bound_method %.50e, %TrivialRange.as.Iterate.impl.NewCursor.93de00.2 [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bound.410: <bound method> = bound_method %TrivialRange.val, %TrivialRange.as.Iterate.impl.NewCursor.93de00.1 [concrete]
// CHECK:STDOUT:   %.d49: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.741, %Iterate.facet.38e [concrete]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.bound: <bound method> = bound_method %.50e, %TrivialRange.as.Iterate.impl.Next [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.e1d, @Optional.HasValue(%Copy.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.fd5, @Optional.Get(%Copy.facet) [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Copy.impl.Op.type: type = fn_type @empty_tuple.type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.Copy.impl.Op: %empty_tuple.type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.5b1: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.23a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f8f)]
// CHECK:STDOUT:   %Core.import_ref.93e: @Optional.%Optional.Get.type (%Optional.Get.type.f7a) = import_ref Core//prelude/parts/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.c2b)]
// CHECK:STDOUT:   %Core.import_ref.9c3: %empty_tuple.type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%empty_tuple.type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.229 = impl_witness_table (%Core.import_ref.9c3), @empty_tuple.type.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.cb1 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_18.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %TrivialRange.ref: type = name_ref TrivialRange, file.%TrivialRange.decl [concrete = constants.%TrivialRange]
// CHECK:STDOUT:   %.loc18_18.2: ref %TrivialRange = temporary_storage
// CHECK:STDOUT:   %.loc18_18.3: init %TrivialRange to %.loc18_18.2 = class_init () [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %.loc18_20.1: init %TrivialRange = converted %.loc18_18.1, %.loc18_18.3 [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %.loc18_20.2: ref %TrivialRange = temporary %.loc18_18.2, %.loc18_20.1 [concrete = constants.%.50e]
// CHECK:STDOUT:   %impl.elem2: %.953 = impl_witness_access constants.%Iterate.impl_witness, element2 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.93de00.2]
// CHECK:STDOUT:   %bound_method.loc18_35.1: <bound method> = bound_method %.loc18_20.2, %impl.elem2 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.bound.464]
// CHECK:STDOUT:   %.loc18_20.3: %TrivialRange = acquire_value %.loc18_20.2 [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %NewCursor.ref: %TrivialRange.as.Iterate.impl.NewCursor.type.de32c1.1 = name_ref NewCursor, @TrivialRange.as.Iterate.impl.%TrivialRange.as.Iterate.impl.NewCursor.decl.loc6_31.1 [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.93de00.1]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.bound: <bound method> = bound_method %.loc18_20.3, %NewCursor.ref [concrete = constants.%TrivialRange.as.Iterate.impl.NewCursor.bound.410]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.NewCursor.call: init %empty_tuple.type = call %TrivialRange.as.Iterate.impl.NewCursor.bound(%.loc18_20.3)
// CHECK:STDOUT:   %var: ref %empty_tuple.type = var_storage invalid
// CHECK:STDOUT:   assign %var, %TrivialRange.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.d49 = impl_witness_access constants.%Iterate.impl_witness, element3 [concrete = constants.%TrivialRange.as.Iterate.impl.Next]
// CHECK:STDOUT:   %bound_method.loc18_35.2: <bound method> = bound_method %.loc18_20.2, %impl.elem3 [concrete = constants.%TrivialRange.as.Iterate.impl.Next.bound]
// CHECK:STDOUT:   %.loc18_35.1: ref %Optional.9a5 = temporary_storage
// CHECK:STDOUT:   %.loc18_20.4: %TrivialRange = acquire_value %.loc18_20.2 [concrete = constants.%TrivialRange.val]
// CHECK:STDOUT:   %TrivialRange.as.Iterate.impl.Next.call: init %Optional.9a5 to %.loc18_35.1 = call %bound_method.loc18_35.2(%.loc18_20.4, %var)
// CHECK:STDOUT:   %.loc18_35.2: ref %Optional.9a5 = temporary %.loc18_35.1, %TrivialRange.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc18_35.3: %Optional.HasValue.type.871 = specific_constant imports.%Core.import_ref.5b1, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.e1d]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.871 = name_ref HasValue, %.loc18_35.3 [concrete = constants.%Optional.HasValue.e1d]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc18_35.2, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%Copy.facet) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.3: <bound method> = bound_method %.loc18_35.2, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc18_35.4: %Optional.9a5 = acquire_value %.loc18_35.2
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc18_35.3(%.loc18_35.4)
// CHECK:STDOUT:   %.loc18_35.5: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc18_35.6: bool = converted %Optional.HasValue.call, %.loc18_35.5
// CHECK:STDOUT:   if %.loc18_35.6 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc18_35.7: %Optional.Get.type.76d = specific_constant imports.%Core.import_ref.93e, @Optional(constants.%Copy.facet) [concrete = constants.%Optional.Get.fd5]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.76d = name_ref Get, %.loc18_35.7 [concrete = constants.%Optional.Get.fd5]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc18_35.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%Copy.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_35.4: <bound method> = bound_method %.loc18_35.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc18_35.8: %Optional.9a5 = acquire_value %.loc18_35.2
// CHECK:STDOUT:   %Optional.Get.call: init %empty_tuple.type = call %bound_method.loc18_35.4(%.loc18_35.8)
// CHECK:STDOUT:   %.loc18_35.9: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc18_35.10: ref %empty_tuple.type = temporary %.loc18_35.9, %Optional.Get.call
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_35.11: %empty_tuple.type = converted %Optional.Get.call, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_12.1: type = splice_block %.loc18_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_12.3: type = converted %.loc18_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %empty_tuple.type = wrapper_binding _, %.loc18_35.11
// CHECK:STDOUT:   %Body.ref: %Body.type = name_ref Body, file.%Body.decl [concrete = constants.%Body]
// CHECK:STDOUT:   %Body.call: init %empty_tuple.type = call %Body.ref()
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %AfterLoop.ref: %AfterLoop.type = name_ref AfterLoop, file.%AfterLoop.decl [concrete = constants.%AfterLoop]
// CHECK:STDOUT:   %AfterLoop.call: init %empty_tuple.type = call %AfterLoop.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_35.1(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_35.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_35.3(%self.param: ref %struct_type.has_value.value.da2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_35.4(%self.param: ref %Optional.9a5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_20.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_20.2(%self.param: ref %TrivialRange) = "no_op";
// CHECK:STDOUT:
