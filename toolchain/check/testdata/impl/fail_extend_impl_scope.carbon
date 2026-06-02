// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_extend_impl_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_extend_impl_scope.carbon

// --- fail_extend_impl_file_scope.carbon
library "[[@TEST_NAME]]";

interface I {}

// CHECK:STDERR: fail_extend_impl_file_scope.carbon:[[@LINE+4]]:1: error: `extend impl` can only be used in an interface or class [ExtendImplOutsideClass]
// CHECK:STDERR: extend impl () as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extend impl () as I {}

// --- fail_extend_impl_function_scope.carbon
library "[[@TEST_NAME]]";

interface J {}

fn F() {
  // CHECK:STDERR: fail_extend_impl_function_scope.carbon:[[@LINE+4]]:3: error: `extend impl` can only be used in an interface or class [ExtendImplOutsideClass]
  // CHECK:STDERR:   extend impl {} as J {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend impl {} as J {}
}

// --- fail_extend_impl_self_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  fn Zero();

  // CHECK:STDERR: fail_extend_impl_self_interface.carbon:[[@LINE+4]]:15: error: `impl as` can only be used in a class [ImplAsOutsideClass]
  // CHECK:STDERR:   extend impl as Z {
  // CHECK:STDERR:               ^~
  // CHECK:STDERR:
  extend impl as Z {
    fn Zero() {}
  }
}

class Point {
  extend impl as Z {
    fn Zero() {}
  }
}

fn F() {
  // Even if the `extend impl` is diagnosed above, we must not add the impl of
  // the interface to itself in a way that allows it to be used during impl
  // lookup, or we end up with infinite impl lookup recursion here.
  ({} as Point).Zero();
}

// CHECK:STDOUT: --- fail_extend_impl_file_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_tuple.type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_14.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_14.2: type = converted %.loc9_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc9_14.2 as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @empty_tuple.type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_impl_function_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @empty_struct_type.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.J.impl: %.loc10_16.2 as %J.ref {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @empty_struct_type.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_16.2: type = converted %.loc10_16.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_impl_self_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.91f: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.WithSelf.Zero.type.205: type = fn_type @Z.WithSelf.Zero, @Z.WithSelf(%Self.91f) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Zero.b63: %Z.WithSelf.Zero.type.205 = struct_value () [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.c9a: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z.WithSelf.Zero.decl [concrete]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type: type = fn_type @<error>.as.Z.impl.Zero, @<error>.as.Z.impl(%Self.91f) [symbolic]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero: %<error>.as.Z.impl.Zero.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @Point.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.type: type = fn_type @Point.as.Z.impl.Zero [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero: %Point.as.Z.impl.Zero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Zero.type.840: type = fn_type @Z.WithSelf.Zero, @Z.WithSelf(%Z.facet) [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Zero.7a6: %Z.WithSelf.Zero.type.840 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [concrete]
// CHECK:STDOUT:   %.27c: ref %Point = temporary invalid, %Point.val [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Point.type.facet: %type = facet_value %Point, () [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Zero.type.63a: type = fn_type @Z.WithSelf.Zero, @Z.WithSelf(%Point.type.facet) [concrete]
// CHECK:STDOUT:   %Z.WithSelf.Zero.777: %Z.WithSelf.Zero.type.63a = struct_value () [concrete]
// CHECK:STDOUT:   %.a52: type = fn_type_with_self_type %Z.WithSelf.Zero.type.840, %Z.facet [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc25_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.27c, %Destroy.Op.2dd605.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Point = %Point.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %Point.decl: type = class_decl @Point [concrete = constants.%Point] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.91f]
// CHECK:STDOUT:   %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Z.WithSelf.Zero.decl: @Z.WithSelf.%Z.WithSelf.Zero.type (%Z.WithSelf.Zero.type.205) = fn_decl @Z.WithSelf.Zero [symbolic = @Z.WithSelf.%Z.WithSelf.Zero (constants.%Z.WithSelf.Zero.b63)] {} {}
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Z.WithSelf.Zero.decl [concrete = constants.%assoc0.c9a]
// CHECK:STDOUT:   impl_decl @<error>.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Zero = @Z.WithSelf.%assoc0
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   witness = (@Z.WithSelf.%Z.WithSelf.Zero.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @<error>.as.Z.impl(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.91f)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type: type = fn_type @<error>.as.Z.impl.Zero, @<error>.as.Z.impl(%Self) [symbolic = %<error>.as.Z.impl.Zero.type (constants.%<error>.as.Z.impl.Zero.type)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero: @<error>.as.Z.impl.%<error>.as.Z.impl.Zero.type (%<error>.as.Z.impl.Zero.type) = struct_value () [symbolic = %<error>.as.Z.impl.Zero (constants.%<error>.as.Z.impl.Zero)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: <error> as %Z.ref {
// CHECK:STDOUT:     %<error>.as.Z.impl.Zero.decl: @<error>.as.Z.impl.%<error>.as.Z.impl.Zero.type (%<error>.as.Z.impl.Zero.type) = fn_decl @<error>.as.Z.impl.Zero [symbolic = @<error>.as.Z.impl.%<error>.as.Z.impl.Zero (constants.%<error>.as.Z.impl.Zero)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Zero = %<error>.as.Z.impl.Zero.decl
// CHECK:STDOUT:     extend %Z.ref
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Point.as.Z.impl: %Self.ref as %Z.ref {
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.decl: %Point.as.Z.impl.Zero.type = fn_decl @Point.as.Z.impl.Zero [concrete = constants.%Point.as.Z.impl.Zero] {} {}
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%Point.as.Z.impl.Zero.decl), @Point.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Point.as.Z.impl.Zero.decl
// CHECK:STDOUT:   extend %Z.ref
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @Point.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Zero = <poisoned>
// CHECK:STDOUT:   extend @Point.as.Z.impl.%Z.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.WithSelf.Zero(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @<error>.as.Z.impl.Zero(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Point.as.Z.impl.Zero() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Point.ref: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %.loc25_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc25_5.3: init %Point to %.loc25_5.2 = class_init () [concrete = constants.%Point.val]
// CHECK:STDOUT:   %.loc25_7.1: init %Point = converted %.loc25_5.1, %.loc25_5.3 [concrete = constants.%Point.val]
// CHECK:STDOUT:   %.loc25_7.2: ref %Point = temporary %.loc25_5.2, %.loc25_7.1 [concrete = constants.%.27c]
// CHECK:STDOUT:   %Zero.ref: %Z.assoc_type = name_ref Zero, @Z.WithSelf.%assoc0 [concrete = constants.%assoc0.c9a]
// CHECK:STDOUT:   %impl.elem0: %.a52 = impl_witness_access constants.%Z.impl_witness, element0 [concrete = constants.%Point.as.Z.impl.Zero]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.27c)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc25_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc25_7.2(%self.param: ref %Point) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.91f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.91f
// CHECK:STDOUT:   %Z.WithSelf.Zero.type => constants.%Z.WithSelf.Zero.type.205
// CHECK:STDOUT:   %Z.WithSelf.Zero => constants.%Z.WithSelf.Zero.b63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Zero(constants.%Self.91f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl(constants.%Self.91f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.91f
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type => constants.%<error>.as.Z.impl.Zero.type
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero => constants.%<error>.as.Z.impl.Zero
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl.Zero(constants.%Self.91f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Z.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Z.facet
// CHECK:STDOUT:   %Z.WithSelf.Zero.type => constants.%Z.WithSelf.Zero.type.840
// CHECK:STDOUT:   %Z.WithSelf.Zero => constants.%Z.WithSelf.Zero.7a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf.Zero(constants.%Z.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Point.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Point.type.facet
// CHECK:STDOUT:   %Z.WithSelf.Zero.type => constants.%Z.WithSelf.Zero.type.63a
// CHECK:STDOUT:   %Z.WithSelf.Zero => constants.%Z.WithSelf.Zero.777
// CHECK:STDOUT: }
// CHECK:STDOUT:
