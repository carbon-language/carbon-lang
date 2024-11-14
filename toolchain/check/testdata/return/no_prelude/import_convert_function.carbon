// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/no_prelude/import_convert_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/no_prelude/import_convert_function.carbon

// --- core.carbon

// A minimized prelude intended to contain as few functions as possible.
package Core;

fn Int32() -> type = "int.make_type_32";

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

// --- library.carbon

package P library "[[@TEST_NAME]]";

import Core;

class C(N:! i32) {}
class D { var n: i32; var m: i32; }
// CHECK:STDERR: library.carbon:[[@LINE+7]]:25: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: fn Make() -> D { return {.n = 0, .m = 0}; }
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE+4]]:25: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn Make() -> D { return {.n = 0, .m = 0}; }
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make() -> D { return {.n = 0, .m = 0}; }

// CHECK:STDERR: library.carbon:[[@LINE+10]]:6: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE+7]]:6: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE-17]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+10]]:6: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE+7]]:6: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE-28]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+10]]:6: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE+7]]:6: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE-39]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+10]]:6: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE+7]]:6: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE-50]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+10]]:6: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE+7]]:6: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE-61]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+10]]:6: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE+7]]:6: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE-72]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+10]]:6: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE+7]]:6: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE-83]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+10]]:6: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE+7]]:6: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~
// CHECK:STDERR: library.carbon:[[@LINE-94]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }

// --- user.carbon

import P library "library";
import Core;

// Check that we don't crash if we import functions and reallocate the function
// value store while handling a `return` statement.
fn F0(n: i32) -> P.D {
  // CHECK:STDERR: user.carbon:[[@LINE+11]]:29: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   if (false) { return {} as P.C(0); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE+8]]:29: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   if (false) { return {} as P.C(0); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE-12]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  if (false) { return {} as P.C(0); }
  // CHECK:STDERR: user.carbon:[[@LINE+11]]:29: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   if (false) { return {} as P.C(1); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE+8]]:29: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   if (false) { return {} as P.C(1); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE-24]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  if (false) { return {} as P.C(1); }
  // CHECK:STDERR: user.carbon:[[@LINE+11]]:29: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   if (false) { return {} as P.C(2); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE+8]]:29: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   if (false) { return {} as P.C(2); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE-36]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  if (false) { return {} as P.C(2); }
  // CHECK:STDERR: user.carbon:[[@LINE+11]]:29: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   if (false) { return {} as P.C(3); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE+8]]:29: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   if (false) { return {} as P.C(3); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE-48]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  if (false) { return {} as P.C(3); }
  // CHECK:STDERR: user.carbon:[[@LINE+11]]:29: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   if (false) { return {} as P.C(4); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE+8]]:29: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   if (false) { return {} as P.C(4); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE-60]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  if (false) { return {} as P.C(4); }
  // CHECK:STDERR: user.carbon:[[@LINE+11]]:29: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   if (false) { return {} as P.C(5); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE+8]]:29: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   if (false) { return {} as P.C(5); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE-72]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  if (false) { return {} as P.C(5); }
  // CHECK:STDERR: user.carbon:[[@LINE+11]]:29: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   if (false) { return {} as P.C(6); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE+8]]:29: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   if (false) { return {} as P.C(6); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE-84]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  if (false) { return {} as P.C(6); }
  // CHECK:STDERR: user.carbon:[[@LINE+10]]:29: error: cannot implicitly convert from `Core.IntLiteral` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   if (false) { return {} as P.C(7); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE+7]]:29: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   if (false) { return {} as P.C(7); }
  // CHECK:STDERR:                             ^~~~
  // CHECK:STDERR: user.carbon:[[@LINE-96]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  if (false) { return {} as P.C(7); }
  return P.Make();
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.1 = interface_decl @ImplicitAs [template = constants.%ImplicitAs] {
// CHECK:STDOUT:     %T.patt.loc7_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_22.1, runtime_param<invalid> [symbolic = %T.patt.loc7_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc7_22.1: type) {
// CHECK:STDOUT:   %T.loc7_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc7_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T.loc7_22.2) [symbolic = %Convert.type (constants.%Convert.type)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type) = struct_value () [symbolic = %Convert (constants.%Convert)]
// CHECK:STDOUT:   %.loc8_32.2: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type) [symbolic = %.loc8_32.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc8_32.3: @ImplicitAs.%.loc8_32.2 (%.1) = assoc_entity element0, %Convert.decl [symbolic = %.loc8_32.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type) = fn_decl @Convert [symbolic = @ImplicitAs.%Convert (constants.%Convert)] {
// CHECK:STDOUT:       %self.patt: @Convert.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.%Self (%Self) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc8_20.1: @Convert.%ImplicitAs.type (%ImplicitAs.type.2) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @Convert.%ImplicitAs.type (%ImplicitAs.type.2) = name_ref Self, %.loc8_20.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc8_20.2: type = facet_type_access %Self.ref [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc8_20.3: type = converted %Self.ref, %.loc8_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc7_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.%Self (%Self) = value_param runtime_param0
// CHECK:STDOUT:       %self: @Convert.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8_32.1: @ImplicitAs.%.loc8_32.2 (%.1) = assoc_entity element0, %Convert.decl [symbolic = %.loc8_32.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %.loc8_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(@ImplicitAs.%T.loc7_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self (%Self)]() -> @Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc7_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%T) {
// CHECK:STDOUT:   %T.loc7_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%T.loc7_22.2) {
// CHECK:STDOUT:   %T.loc7_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %D, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.n: i32, .m: i32} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.13: %.9 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.1, @ImplicitAs(%D) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.4 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.16: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.17: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.18: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.19: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6 [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %.20: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.7 [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.21: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.8 [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %.22: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.9 [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//default, inst+5, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//default, inst+15, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//default, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.9) = import_ref Core//default, inst+43, loaded [symbolic = @ImplicitAs.%.2 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.5: @ImplicitAs.%Convert.type (%Convert.type.1) = import_ref Core//default, inst+36, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %N.patt.loc6_9.1: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc6_9.1, runtime_param<invalid> [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_13.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc6_13.2: type = converted %int.make_type_32, %.loc6_13.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_9.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc27_8: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:     %.loc27_7: i32 = converted %.loc27_8, <error> [template = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc38_8: Core.IntLiteral = int_value 1 [template = constants.%.16]
// CHECK:STDOUT:     %.loc38_7: i32 = converted %.loc38_8, <error> [template = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc49_8: Core.IntLiteral = int_value 2 [template = constants.%.17]
// CHECK:STDOUT:     %.loc49_7: i32 = converted %.loc49_8, <error> [template = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc60_8: Core.IntLiteral = int_value 3 [template = constants.%.18]
// CHECK:STDOUT:     %.loc60_7: i32 = converted %.loc60_8, <error> [template = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc71_8: Core.IntLiteral = int_value 4 [template = constants.%.19]
// CHECK:STDOUT:     %.loc71_7: i32 = converted %.loc71_8, <error> [template = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc82_8: Core.IntLiteral = int_value 5 [template = constants.%.20]
// CHECK:STDOUT:     %.loc82_7: i32 = converted %.loc82_8, <error> [template = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc93_8: Core.IntLiteral = int_value 6 [template = constants.%.21]
// CHECK:STDOUT:     %.loc93_7: i32 = converted %.loc93_8, <error> [template = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc104_8: Core.IntLiteral = int_value 7 [template = constants.%.22]
// CHECK:STDOUT:     %.loc104_7: i32 = converted %.loc104_8, <error> [template = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, imports.%import_ref.2 [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.9) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.3 = fn_decl @Convert.2 [template = constants.%Convert.3] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_33: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc27_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.5 = fn_decl @Convert.3 [template = constants.%Convert.5] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc38_33: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc38_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.6 = fn_decl @Convert.4 [template = constants.%Convert.6] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc49_33: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc49_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.7 = fn_decl @Convert.5 [template = constants.%Convert.7] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc60_33: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc60_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.8 = fn_decl @Convert.6 [template = constants.%Convert.8] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc71_33: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc71_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.9 = fn_decl @Convert.7 [template = constants.%Convert.9] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc82_33: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc82_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.10 = fn_decl @Convert.8 [template = constants.%Convert.10] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc93_33: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc93_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.11 = fn_decl @Convert.9 [template = constants.%Convert.11] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc104_33: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc104_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc6_9.1: i32) {
// CHECK:STDOUT:   %N.loc6_9.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6_19: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %int.make_type_32.loc7_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_18.1: type = value_of_initializer %int.make_type_32.loc7_18 [template = i32]
// CHECK:STDOUT:   %.loc7_18.2: type = converted %int.make_type_32.loc7_18, %.loc7_18.1 [template = i32]
// CHECK:STDOUT:   %.loc7_16: %.3 = field_decl n, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc7_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_30.1: type = value_of_initializer %int.make_type_32.loc7_30 [template = i32]
// CHECK:STDOUT:   %.loc7_30.2: type = converted %int.make_type_32.loc7_30, %.loc7_30.1 [template = i32]
// CHECK:STDOUT:   %.loc7_28: %.3 = field_decl m, element1 [template]
// CHECK:STDOUT:   %.loc7_35: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc7_16
// CHECK:STDOUT:   .m = %.loc7_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_31: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_39: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_40.1: %.8 = struct_literal (%.loc15_31, %.loc15_39)
// CHECK:STDOUT:   %.loc15_40.2: i32 = converted %.loc15_31, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: <error>]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc27: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc27
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: <error>]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc38: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc38
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: <error>]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc49: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc49
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: <error>]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc60: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc60
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: <error>]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc71: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc71
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: <error>]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc82: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc82
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: <error>]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc93: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc93
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: <error>]() -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc104: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc104
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:   %T.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %T.patt => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %.2 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%D, <error>) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {.n: i32, .m: i32} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [template]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.4: bool = bool_literal false [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %.12: %.8 = assoc_entity element0, imports.%import_ref.13 [symbolic]
// CHECK:STDOUT:   %.13: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.14: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.15: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.16: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.17: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.18: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.19: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [template] {
// CHECK:STDOUT:     .D = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.6
// CHECK:STDOUT:     .Make = %import_ref.14
// CHECK:STDOUT:     import P//library
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.8
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//default, inst+5, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: type = import_ref P//library, inst+29, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.3 = import_ref P//library, inst+30, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref P//library, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref P//library, inst+39, unloaded
// CHECK:STDOUT:   %import_ref.6: %C.type = import_ref P//library, inst+20, loaded [template = constants.%C.1]
// CHECK:STDOUT:   %import_ref.7 = import_ref P//library, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.8: %ImplicitAs.type.1 = import_ref Core//default, inst+15, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//default, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.10: @ImplicitAs.%.1 (%.8) = import_ref Core//default, inst+43, loaded [symbolic = @ImplicitAs.%.2 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//default, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.14: %Make.type = import_ref P//library, inst+48, loaded [template = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .P = imports.%P
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F0 = %F0.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F0.decl: %F0.type = fn_decl @F0 [template = constants.%F0] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc7_10.2: type = converted %int.make_type_32, %.loc7_10.1 [template = i32]
// CHECK:STDOUT:     %P.ref.loc7: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.2 [template = constants.%D]
// CHECK:STDOUT:     %n.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.8) = assoc_entity element0, imports.%import_ref.12 [symbolic = %.2 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9
// CHECK:STDOUT:     .Convert = imports.%import_ref.10
// CHECK:STDOUT:     witness = (imports.%import_ref.11)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .n = imports.%import_ref.4
// CHECK:STDOUT:   .m = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%N: i32) {
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param_patt: i32) -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc19_7 br !if.then.loc19 else br !if.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc19:
// CHECK:STDOUT:   %.loc19_24: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc19: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc19: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc19_33: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc19_32: i32 = converted %.loc19_33, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc19:
// CHECK:STDOUT:   %.loc31_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc31_7 br !if.then.loc31 else br !if.else.loc31
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc31:
// CHECK:STDOUT:   %.loc31_24: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc31: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc31: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc31_33: Core.IntLiteral = int_value 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc31_32: i32 = converted %.loc31_33, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc31:
// CHECK:STDOUT:   %.loc43_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc43_7 br !if.then.loc43 else br !if.else.loc43
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc43:
// CHECK:STDOUT:   %.loc43_24: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc43: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc43: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc43_33: Core.IntLiteral = int_value 2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc43_32: i32 = converted %.loc43_33, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc43:
// CHECK:STDOUT:   %.loc55_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc55_7 br !if.then.loc55 else br !if.else.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc55:
// CHECK:STDOUT:   %.loc55_24: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc55: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc55: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc55_33: Core.IntLiteral = int_value 3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc55_32: i32 = converted %.loc55_33, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc55:
// CHECK:STDOUT:   %.loc67_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc67_7 br !if.then.loc67 else br !if.else.loc67
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc67:
// CHECK:STDOUT:   %.loc67_24: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc67: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc67: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc67_33: Core.IntLiteral = int_value 4 [template = constants.%.16]
// CHECK:STDOUT:   %.loc67_32: i32 = converted %.loc67_33, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc67:
// CHECK:STDOUT:   %.loc79_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc79_7 br !if.then.loc79 else br !if.else.loc79
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc79:
// CHECK:STDOUT:   %.loc79_24: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc79: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc79: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc79_33: Core.IntLiteral = int_value 5 [template = constants.%.17]
// CHECK:STDOUT:   %.loc79_32: i32 = converted %.loc79_33, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc79:
// CHECK:STDOUT:   %.loc91_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc91_7 br !if.then.loc91 else br !if.else.loc91
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc91:
// CHECK:STDOUT:   %.loc91_24: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc91: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc91: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc91_33: Core.IntLiteral = int_value 6 [template = constants.%.18]
// CHECK:STDOUT:   %.loc91_32: i32 = converted %.loc91_33, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc91:
// CHECK:STDOUT:   %.loc102_7: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc102_7 br !if.then.loc102 else br !if.else.loc102
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc102:
// CHECK:STDOUT:   %.loc102_24: %.5 = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc102: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %C.ref.loc102: %C.type = name_ref C, imports.%import_ref.6 [template = constants.%C.1]
// CHECK:STDOUT:   %.loc102_33: Core.IntLiteral = int_value 7 [template = constants.%.19]
// CHECK:STDOUT:   %.loc102_32: i32 = converted %.loc102_33, <error> [template = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc102:
// CHECK:STDOUT:   %P.ref.loc103: <namespace> = name_ref P, imports.%P [template = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%import_ref.14 [template = constants.%Make]
// CHECK:STDOUT:   %.loc7_15.2: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc7_15.2
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%T: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self (%Self.2)]() -> @Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:   %T.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
