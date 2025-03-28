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

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl IntLiteral() as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

// --- library.carbon

package P library "[[@TEST_NAME]]";

import Core;

class C(N:! i32) {}
class D { var n: i32; var m: i32; }
// CHECK:STDERR: library.carbon:[[@LINE+8]]:25: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn Make() -> D { return {.n = 0, .m = 0}; }
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make() -> D { return {.n = 0, .m = 0}; }

// CHECK:STDERR: library.carbon:[[@LINE+11]]:6: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: library.carbon:[[@LINE-17]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-19]]:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+11]]:6: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: library.carbon:[[@LINE-29]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-31]]:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+11]]:6: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: library.carbon:[[@LINE-41]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-43]]:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+11]]:6: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: library.carbon:[[@LINE-53]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-55]]:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+11]]:6: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: library.carbon:[[@LINE-65]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-67]]:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+11]]:6: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: library.carbon:[[@LINE-77]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-79]]:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+11]]:6: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: library.carbon:[[@LINE-89]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-91]]:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR: library.carbon:[[@LINE+11]]:6: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
// CHECK:STDERR:      ^~~~
// CHECK:STDERR: library.carbon:[[@LINE-101]]:1: in import [InImport]
// CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface ImplicitAs(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: library.carbon:[[@LINE-103]]:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(N:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }

// --- user.carbon

import P library "library";
import Core;

// Check that we don't crash if we import functions and reallocate the function
// value store while handling a `return` statement.
fn F0(n: i32) -> P.D {
  // CHECK:STDERR: user.carbon:[[@LINE+12]]:29: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   if (false) { return {} as P.C(0); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-8]]:1: in import [InImport]
  // CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-13]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(0); }
  // CHECK:STDERR: user.carbon:[[@LINE+12]]:29: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   if (false) { return {} as P.C(1); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-21]]:1: in import [InImport]
  // CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-26]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(1); }
  // CHECK:STDERR: user.carbon:[[@LINE+12]]:29: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   if (false) { return {} as P.C(2); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-34]]:1: in import [InImport]
  // CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-39]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(2); }
  // CHECK:STDERR: user.carbon:[[@LINE+12]]:29: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   if (false) { return {} as P.C(3); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-47]]:1: in import [InImport]
  // CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-52]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(3); }
  // CHECK:STDERR: user.carbon:[[@LINE+12]]:29: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   if (false) { return {} as P.C(4); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-60]]:1: in import [InImport]
  // CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-65]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(4); }
  // CHECK:STDERR: user.carbon:[[@LINE+12]]:29: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   if (false) { return {} as P.C(5); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-73]]:1: in import [InImport]
  // CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-78]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(5); }
  // CHECK:STDERR: user.carbon:[[@LINE+12]]:29: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   if (false) { return {} as P.C(6); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-86]]:1: in import [InImport]
  // CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-91]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(6); }
  // CHECK:STDERR: user.carbon:[[@LINE+12]]:29: error: member access into undefined facet type `Core.ImplicitAs(<builtin i32>)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   if (false) { return {} as P.C(7); }
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-99]]:1: in import [InImport]
  // CHECK:STDERR: core.carbon:8:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon:[[@LINE-104]]:1: in import [InImport]
  // CHECK:STDERR: library.carbon:6:9: note: initializing generic parameter `N` declared here [InitializingGenericParam]
  // CHECK:STDERR: class C(N:! i32) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (false) { return {} as P.C(7); }
  return P.Make();
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.94e: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type.94e = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.11a: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Convert.type.752: type = fn_type @Convert.1, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %Convert.fcc: %Convert.type.752 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.dd3: type = assoc_entity_type %ImplicitAs.type.11a [concrete]
// CHECK:STDOUT:   %assoc0.7cc: %ImplicitAs.assoc_type.dd3 = assoc_entity element0, @ImplicitAs.%Convert.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.c2a: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.40d: %Convert.type.c2a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.11a = facet_value Core.IntLiteral, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc6_22.1: type = splice_block %.loc6_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_22.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_22.3: type = converted %int_literal.make_type, %.loc6_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc8_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc12_17.1: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc12_17.2: type = converted %int_literal.make_type, %.loc12_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc12_36.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc12_36.2: type = converted %int.make_type_signed, %.loc12_36.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.11a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc8_22.1: type) {
// CHECK:STDOUT:   %T.loc8_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc8_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T.loc8_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.94e)]
// CHECK:STDOUT:   %assoc0.loc9_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.94e) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_32.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc8_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_20.1: type = splice_block %.loc9_20.3 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc9_20.2: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc9_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc9_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc9_20.3: type = converted %Self.ref, %Self.as_type.loc9_20.2 [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc9_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.94e) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc9_32.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc9_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc12_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.c2a = fn_decl @Convert.2 [concrete = constants.%Convert.40d] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_signed, %.loc13_31.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc12_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@ImplicitAs.%T.loc8_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc9_20.1 (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc8_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.752
// CHECK:STDOUT:   %Convert => constants.%Convert.fcc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.dd3
// CHECK:STDOUT:   %assoc0.loc9_32.2 => constants.%assoc0.7cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32.builtin, constants.%ImplicitAs.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet
// CHECK:STDOUT:   %Self.as_type.loc9_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32.builtin [concrete]
// CHECK:STDOUT:   %struct_type.n.m.566: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [concrete]
// CHECK:STDOUT:   %complete_type.682: <witness> = complete_type_witness %struct_type.n.m.566 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.94e: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %Convert.type.e635b7.1: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.50bb84.1: %Convert.type.e635b7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.type.e635b7.2: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.50bb84.2: %Convert.type.e635b7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.type.e635b7.3: type = fn_type @Convert.4 [concrete]
// CHECK:STDOUT:   %Convert.50bb84.3: %Convert.type.e635b7.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.type.e635b7.4: type = fn_type @Convert.5 [concrete]
// CHECK:STDOUT:   %Convert.50bb84.4: %Convert.type.e635b7.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.type.e635b7.5: type = fn_type @Convert.6 [concrete]
// CHECK:STDOUT:   %Convert.50bb84.5: %Convert.type.e635b7.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.type.e635b7.6: type = fn_type @Convert.7 [concrete]
// CHECK:STDOUT:   %Convert.50bb84.6: %Convert.type.e635b7.6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.type.e635b7.7: type = fn_type @Convert.8 [concrete]
// CHECK:STDOUT:   %Convert.50bb84.7: %Convert.type.e635b7.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.type.e635b7.8: type = fn_type @Convert.9 [concrete]
// CHECK:STDOUT:   %Convert.50bb84.8: %Convert.type.e635b7.8 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc8_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst48 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.31f = import_ref Core//default, loc9_32, unloaded
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc8_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst48 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//default, loc9_32, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %N.patt.loc6_9.1: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_13.1: type = splice_block %.loc6_13.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc6_13.2: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc6_13.3: type = converted %int.make_type_signed, %.loc6_13.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.1: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.b6e97d.1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc29: %i32.builtin = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.b6e97d.2 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %.loc41: %i32.builtin = converted %int_1, <error> [concrete = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.b6e97d.3 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %.loc53: %i32.builtin = converted %int_2, <error> [concrete = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.b6e97d.4 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc65: %i32.builtin = converted %int_3, <error> [concrete = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.b6e97d.5 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %.loc77: %i32.builtin = converted %int_4, <error> [concrete = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.b6e97d.6 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:     %.loc89: %i32.builtin = converted %int_5, <error> [concrete = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.b6e97d.7 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:     %.loc101: %i32.builtin = converted %int_6, <error> [concrete = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.b6e97d.8 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:     %.loc113: %i32.builtin = converted %int_7, <error> [concrete = <error>]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, imports.%Core.import_ref.207 [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.31f
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b6e97d.1: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e635b7.1 = fn_decl @Convert.2 [concrete = constants.%Convert.50bb84.1] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b6e97d.2: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e635b7.2 = fn_decl @Convert.3 [concrete = constants.%Convert.50bb84.2] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b6e97d.3: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e635b7.3 = fn_decl @Convert.4 [concrete = constants.%Convert.50bb84.3] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b6e97d.4: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e635b7.4 = fn_decl @Convert.5 [concrete = constants.%Convert.50bb84.4] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b6e97d.5: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e635b7.5 = fn_decl @Convert.6 [concrete = constants.%Convert.50bb84.5] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b6e97d.6: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e635b7.6 = fn_decl @Convert.7 [concrete = constants.%Convert.50bb84.6] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b6e97d.7: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e635b7.7 = fn_decl @Convert.8 [concrete = constants.%Convert.50bb84.7] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b6e97d.8: <error> as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.e635b7.8 = fn_decl @Convert.9 [concrete = constants.%Convert.50bb84.8] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: <error> = value_param call_param0
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc6_9.1: %i32.builtin) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc7_16: %D.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc7_11: %D.elem = var_pattern %.loc7_16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc7_11: ref %D.elem = var <none>
// CHECK:STDOUT:   %.loc7_28: %D.elem = field_decl m, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc7_23: %D.elem = var_pattern %.loc7_28
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc7_23: ref %D.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.m.566 [concrete = constants.%complete_type.682]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc7_16
// CHECK:STDOUT:   .m = %.loc7_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc16_31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_0.loc16_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc16_40.1: %struct_type.n.m.819 = struct_literal (%int_0.loc16_31, %int_0.loc16_39)
// CHECK:STDOUT:   %.loc16_40.2: %i32.builtin = converted %int_0.loc16_31, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: <error>]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc29: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc29
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: <error>]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc41: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc41
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: <error>]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc53: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc53
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: <error>]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc65: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc65
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6[%self.param_patt: <error>]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc77: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc77
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7[%self.param_patt: <error>]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc89: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc89
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8[%self.param_patt: <error>]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc101: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc101
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9[%self.param_patt: <error>]() -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc113: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc113
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
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%D) {
// CHECK:STDOUT:   %T => constants.%D
// CHECK:STDOUT:   %T.patt => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32.builtin, .m: %i32.builtin} [concrete]
// CHECK:STDOUT:   %complete_type.682: <witness> = complete_type_witness %struct_type.n.m [concrete]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [concrete]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [concrete] {
// CHECK:STDOUT:     .D = %P.D
// CHECK:STDOUT:     .C = %P.C
// CHECK:STDOUT:     .Make = %P.Make
// CHECK:STDOUT:     import P//library
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.D: type = import_ref P//library, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %P.import_ref.7e5: <witness> = import_ref P//library, loc7_35, loaded [concrete = constants.%complete_type.682]
// CHECK:STDOUT:   %P.import_ref.cab = import_ref P//library, inst46 [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.a99 = import_ref P//library, loc7_16, unloaded
// CHECK:STDOUT:   %P.import_ref.9d2 = import_ref P//library, loc7_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.85e: %i32.builtin = import_ref P//library, loc6_9, loaded [symbolic = @C.%N (constants.%N)]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc6_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.d9b = import_ref P//library, inst41 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc8_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst48 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.31f = import_ref Core//default, loc9_32, unloaded
// CHECK:STDOUT:   %Core.Convert = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc8_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst48 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//default, loc9_32, unloaded
// CHECK:STDOUT:   %P.Make: %Make.type = import_ref P//library, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .P = imports.%P
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F0 = %F0.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F0.decl: %F0.type = fn_decl @F0 [concrete = constants.%F0] {
// CHECK:STDOUT:     %n.patt: %i32.builtin = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32.builtin = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc7: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %n.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %.loc7_10.1: type = splice_block %.loc7_10.3 [concrete = constants.%i32.builtin] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %int.make_type_signed: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc7_10.2: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:       %.loc7_10.3: type = converted %int.make_type_signed, %.loc7_10.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32.builtin = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type) = struct_value () [symbolic = %Convert (constants.%Convert)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, imports.%Core.import_ref.207 [symbolic = %assoc0 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.31f
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.7e5
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.cab
// CHECK:STDOUT:   .n = imports.%P.import_ref.a99
// CHECK:STDOUT:   .m = imports.%P.import_ref.9d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.85e: %i32.builtin) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32.builtin = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: %i32.builtin = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.d9b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param_patt: %i32.builtin) -> %return.param_patt: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc20: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc20 br !if.then.loc20 else br !if.else.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc20:
// CHECK:STDOUT:   %.loc20_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc20: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc20: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc20_34: %i32.builtin = converted %int_0, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc20:
// CHECK:STDOUT:   %false.loc33: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc33 br !if.then.loc33 else br !if.else.loc33
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc33:
// CHECK:STDOUT:   %.loc33_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc33: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc33: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc33_34: %i32.builtin = converted %int_1, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc33:
// CHECK:STDOUT:   %false.loc46: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc46 br !if.then.loc46 else br !if.else.loc46
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc46:
// CHECK:STDOUT:   %.loc46_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc46: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc46: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc46_34: %i32.builtin = converted %int_2, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc46:
// CHECK:STDOUT:   %false.loc59: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc59 br !if.then.loc59 else br !if.else.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc59:
// CHECK:STDOUT:   %.loc59_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc59: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc59: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc59_34: %i32.builtin = converted %int_3, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc59:
// CHECK:STDOUT:   %false.loc72: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc72 br !if.then.loc72 else br !if.else.loc72
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc72:
// CHECK:STDOUT:   %.loc72_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc72: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc72: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc72_34: %i32.builtin = converted %int_4, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc72:
// CHECK:STDOUT:   %false.loc85: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc85 br !if.then.loc85 else br !if.else.loc85
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc85:
// CHECK:STDOUT:   %.loc85_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc85: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc85: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc85_34: %i32.builtin = converted %int_5, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc85:
// CHECK:STDOUT:   %false.loc98: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc98 br !if.then.loc98 else br !if.else.loc98
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc98:
// CHECK:STDOUT:   %.loc98_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc98: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc98: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc98_34: %i32.builtin = converted %int_6, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc98:
// CHECK:STDOUT:   %false.loc111: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc111 br !if.then.loc111 else br !if.else.loc111
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc111:
// CHECK:STDOUT:   %.loc111_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc111: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc111: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc111_34: %i32.builtin = converted %int_7, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc111:
// CHECK:STDOUT:   %P.ref.loc112: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc7_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc7_15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %D [from "library.carbon"];
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
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT: }
// CHECK:STDOUT:
