// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/fail_extend_impl_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/fail_extend_impl_scope.carbon

// --- fail_extend_impl_file_scope.carbon
library "[[@TEST_NAME]]";

interface I {}

// CHECK:STDERR: fail_extend_impl_file_scope.carbon:[[@LINE+4]]:1: error: `extend impl` can only be used in a class [ExtendImplOutsideClass]
// CHECK:STDERR: extend impl () as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extend impl () as I {}

// --- fail_extend_impl_function_scope.carbon
library "[[@TEST_NAME]]";

interface J {}

fn F() {
  // CHECK:STDERR: fail_extend_impl_function_scope.carbon:[[@LINE+4]]:3: error: `extend impl` can only be used in a class [ExtendImplOutsideClass]
  // CHECK:STDERR:   extend impl {} as J {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend impl {} as J {}
}

// --- fail_extend_impl_self_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  fn Zero();

  // CHECK:STDERR: fail_extend_impl_self_interface.carbon:[[@LINE+11]]:15: error: `impl as` can only be used in a class [ImplAsOutsideClass]
  // CHECK:STDERR:   extend impl as Z {
  // CHECK:STDERR:               ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_extend_impl_self_interface.carbon:[[@LINE+7]]:3: error: impl as incomplete facet type `Z` [ImplAsIncompleteFacetType]
  // CHECK:STDERR:   extend impl as Z {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_extend_impl_self_interface.carbon:[[@LINE-10]]:1: note: interface is currently being defined [InterfaceUndefinedWithinDefinition]
  // CHECK:STDERR: interface Z {
  // CHECK:STDERR: ^~~~~~~~~~~~~
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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_14.2: type = converted %.loc9_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc9_14.2 as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_impl_function_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc10_16.2 as %J.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_16.2: type = converted %.loc10_16.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_impl_self_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Zero.type.822: type = fn_type @Zero.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Zero.d14: %Zero.type.822 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type %Z.type [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%Zero.decl [concrete]
// CHECK:STDOUT:   %Zero.type.db4: type = fn_type @Zero.2, @impl.6b5(%Self) [symbolic]
// CHECK:STDOUT:   %Zero.8fb: %Zero.type.db4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.c94.%Zero.decl) [concrete]
// CHECK:STDOUT:   %Zero.type.e33: type = fn_type @Zero.3 [concrete]
// CHECK:STDOUT:   %Zero.dec: %Zero.type.e33 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, (%impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [concrete]
// CHECK:STDOUT:   %.728: type = fn_type_with_self_type %Zero.type.822, %Z.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Point = %Point.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %Point.decl: type = class_decl @Point [concrete = constants.%Point] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Zero.decl: %Zero.type.822 = fn_decl @Zero.1 [concrete = constants.%Zero.d14] {} {}
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Zero.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   impl_decl @impl.6b5 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Zero = %assoc0
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   witness = (%Zero.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6b5(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Zero.type: type = fn_type @Zero.2, @impl.6b5(%Self) [symbolic = %Zero.type (constants.%Zero.type.db4)]
// CHECK:STDOUT:   %Zero: @impl.6b5.%Zero.type (%Zero.type.db4) = struct_value () [symbolic = %Zero (constants.%Zero.8fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Z.ref {
// CHECK:STDOUT:     %Zero.decl: @impl.6b5.%Zero.type (%Zero.type.db4) = fn_decl @Zero.2 [symbolic = @impl.6b5.%Zero (constants.%Zero.8fb)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Zero = %Zero.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c94: %Self.ref as %Z.ref {
// CHECK:STDOUT:   %Zero.decl: %Zero.type.e33 = fn_decl @Zero.3 [concrete = constants.%Zero.dec] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Zero.decl
// CHECK:STDOUT:   witness = @Point.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @impl.c94 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.c94.%Zero.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Zero = <poisoned>
// CHECK:STDOUT:   extend @impl.c94.%Z.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Zero.1(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Zero.2(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Zero.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc32_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Point.ref: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %.loc32_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc32_5.3: init %Point = class_init (), %.loc32_5.2 [concrete = constants.%Point.val]
// CHECK:STDOUT:   %.loc32_5.4: ref %Point = temporary %.loc32_5.2, %.loc32_5.3
// CHECK:STDOUT:   %.loc32_7: ref %Point = converted %.loc32_5.1, %.loc32_5.4
// CHECK:STDOUT:   %Zero.ref: %Z.assoc_type = name_ref Zero, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.728 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Zero.dec]
// CHECK:STDOUT:   %Zero.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6b5(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Zero.type => constants.%Zero.type.db4
// CHECK:STDOUT:   %Zero => constants.%Zero.8fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6b5(%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Zero.1(constants.%Z.facet) {}
// CHECK:STDOUT:
