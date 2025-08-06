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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
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
// CHECK:STDOUT:     %.loc9_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_14.2: type = converted %.loc9_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @empty_tuple.type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
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
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc9_14.2 as %I.ref {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @F.%J.impl_witness_table [concrete]
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
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.J.impl: %.loc10_16.2 as %J.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_16.2: type = converted %.loc10_16.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @empty_struct_type.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_impl_self_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.6e6: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.Zero.type: type = fn_type @Z.Zero [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.Zero: %Z.Zero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.534: %Z.assoc_type = assoc_entity element0, @Z.%Z.Zero.decl [concrete]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type: type = fn_type @<error>.as.Z.impl.Zero, @<error>.as.Z.impl(%Self.6e6) [symbolic]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero: %<error>.as.Z.impl.Zero.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Point: type = class_type @Point [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @Point.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.type: type = fn_type @Point.as.Z.impl.Zero [concrete]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero: %Point.as.Z.impl.Zero.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %Point, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.e19: <witness> = impl_witness @Point.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.536: type = ptr_type %Point [concrete]
// CHECK:STDOUT:   %pattern_type.bbc: type = pattern_type %ptr.536 [concrete]
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op.type: type = fn_type @Point.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op: %Point.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Point.val: %Point = struct_value () [concrete]
// CHECK:STDOUT:   %.766: type = fn_type_with_self_type %Z.Zero.type, %Z.facet [concrete]
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
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.6e6]
// CHECK:STDOUT:   %Z.Zero.decl: %Z.Zero.type = fn_decl @Z.Zero [concrete = constants.%Z.Zero] {} {}
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, %Z.Zero.decl [concrete = constants.%assoc0.534]
// CHECK:STDOUT:   impl_decl @<error>.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Zero = %assoc0
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   witness = (%Z.Zero.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @<error>.as.Z.impl(@Z.%Self: %Z.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.6e6)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type: type = fn_type @<error>.as.Z.impl.Zero, @<error>.as.Z.impl(%Self) [symbolic = %<error>.as.Z.impl.Zero.type (constants.%<error>.as.Z.impl.Zero.type)]
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero: @<error>.as.Z.impl.%<error>.as.Z.impl.Zero.type (%<error>.as.Z.impl.Zero.type) = struct_value () [symbolic = %<error>.as.Z.impl.Zero (constants.%<error>.as.Z.impl.Zero)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: <error> as %Z.ref {
// CHECK:STDOUT:     %<error>.as.Z.impl.Zero.decl: @<error>.as.Z.impl.%<error>.as.Z.impl.Zero.type (%<error>.as.Z.impl.Zero.type) = fn_decl @<error>.as.Z.impl.Zero [symbolic = @<error>.as.Z.impl.%<error>.as.Z.impl.Zero (constants.%<error>.as.Z.impl.Zero)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Zero = %<error>.as.Z.impl.Zero.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Point.as.Z.impl: %Self.ref as %Z.ref {
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.decl: %Point.as.Z.impl.Zero.type = fn_decl @Point.as.Z.impl.Zero [concrete = constants.%Point.as.Z.impl.Zero] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Zero = %Point.as.Z.impl.Zero.decl
// CHECK:STDOUT:   witness = @Point.%Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Point.as.Destroy.impl: constants.%Point as constants.%Destroy.type {
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op.decl: %Point.as.Destroy.impl.Op.type = fn_decl @Point.as.Destroy.impl.Op [concrete = constants.%Point.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bbc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bbc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.536 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %self: %ptr.536 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Point.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Point.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Point {
// CHECK:STDOUT:   impl_decl @Point.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Point [concrete = constants.%Point]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (@Point.as.Z.impl.%Point.as.Z.impl.Zero.decl), @Point.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:   impl_decl @Point.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Point.as.Destroy.impl.%Point.as.Destroy.impl.Op.decl), @Point.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.e19]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Point
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Zero = <poisoned>
// CHECK:STDOUT:   extend @Point.as.Z.impl.%Z.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Z.Zero(@Z.%Self: %Z.type) {
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
// CHECK:STDOUT: fn @Point.as.Destroy.impl.Op(%self.param: %ptr.536) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Point.ref: type = name_ref Point, file.%Point.decl [concrete = constants.%Point]
// CHECK:STDOUT:   %.loc25_5.2: ref %Point = temporary_storage
// CHECK:STDOUT:   %.loc25_5.3: init %Point = class_init (), %.loc25_5.2 [concrete = constants.%Point.val]
// CHECK:STDOUT:   %.loc25_5.4: ref %Point = temporary %.loc25_5.2, %.loc25_5.3
// CHECK:STDOUT:   %.loc25_7: ref %Point = converted %.loc25_5.1, %.loc25_5.4
// CHECK:STDOUT:   %Zero.ref: %Z.assoc_type = name_ref Zero, @Z.%assoc0 [concrete = constants.%assoc0.534]
// CHECK:STDOUT:   %impl.elem0: %.766 = impl_witness_access constants.%Z.impl_witness, element0 [concrete = constants.%Point.as.Z.impl.Zero]
// CHECK:STDOUT:   %Point.as.Z.impl.Zero.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc25_5.2, constants.%Point.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.536 = addr_of %.loc25_5.2
// CHECK:STDOUT:   %Point.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Point.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Zero(constants.%Self.6e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl(constants.%Self.6e6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.6e6
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero.type => constants.%<error>.as.Z.impl.Zero.type
// CHECK:STDOUT:   %<error>.as.Z.impl.Zero => constants.%<error>.as.Z.impl.Zero
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @<error>.as.Z.impl.Zero(constants.%Self.6e6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.Zero(constants.%Z.facet) {}
// CHECK:STDOUT:
