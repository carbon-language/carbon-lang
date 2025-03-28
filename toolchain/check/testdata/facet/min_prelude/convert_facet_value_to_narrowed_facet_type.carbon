// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_to_narrowed_facet_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/convert_facet_value_to_narrowed_facet_type.carbon

// --- convert_to_narrowed_facet_type.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}

fn Feed[T:! Eats](e: T) {}

// CHECK:STDERR: convert_to_narrowed_facet_type.carbon:[[@LINE+7]]:21: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn HandleAnimal[U:! Animal & Eats](a: U) { Feed(a); }
// CHECK:STDERR:                     ^~~~~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn HandleAnimal[U:! Animal & Eats](a: U) { Feed(a); }

// --- bigger.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}
interface Tame {}

// CHECK:STDERR: bigger.carbon:[[@LINE+7]]:17: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn FeedTame[V:! Tame & Eats](v: V) {}
// CHECK:STDERR:                 ^~~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn FeedTame[V:! Tame & Eats](v: V) {}

// CHECK:STDERR: bigger.carbon:[[@LINE+14]]:25: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn HandleTameAnimal[W:! Eats & Animal & Tame](w: W) {
// CHECK:STDERR:                         ^~~~~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: bigger.carbon:[[@LINE+7]]:25: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn HandleTameAnimal[W:! Eats & Animal & Tame](w: W) {
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn HandleTameAnimal[W:! Eats & Animal & Tame](w: W) {
  FeedTame(w);
}

// --- with_blanket.carbon
library "[[@TEST_NAME]]";

interface Eats {}
interface Animal {}
interface Tame {}

impl forall [A:! Animal] A as Eats {}

// CHECK:STDERR: with_blanket.carbon:[[@LINE+7]]:18: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn FeedTame2[V:! Tame & Eats](v: V) {}
// CHECK:STDERR:                  ^~~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn FeedTame2[V:! Tame & Eats](v: V) {}

// CHECK:STDERR: with_blanket.carbon:[[@LINE+7]]:26: error: member access into undefined facet type `Core.BitAnd` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: fn HandleTameAnimal2[W:! Animal & Tame](w: W) {
// CHECK:STDERR:                          ^~~~~~~~~~~~~
// CHECK:STDERR: include_files/facet_types.carbon:17:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface BitAnd {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn HandleTameAnimal2[W:! Animal & Tame](w: W) {
  FeedTame2(w);
}

// --- equivalent.carbon
library "[[@TEST_NAME]]";

// This is testing `FindWitnessInFacet` from impl_lookup.cpp

interface A {}

fn TakesA[T:! A](x: T) {}

fn WithExtraWhere[U:! A where .Self impls type](y: U) {
  TakesA(y);
}

// --- no_interfaces_success.carbon
library "[[@TEST_NAME]]";

fn TakesTypeDeduced[T:! type](x: T) {}
fn CallsWithExtraWhere[U:! type where .Self impls type](y: U) {
  TakesTypeDeduced(y);
}

fn TakesTypeExplicit(T:! type) {}
fn CallsWithExtraWhereExplicit(U:! type where .Self impls type) {
  TakesTypeExplicit(U);
}

// --- fail_todo_no_interfaces.carbon
library "[[@TEST_NAME]]";

fn TakesExtraWhereDeduced[T:! type where .Self impls type](x: T) {}
fn CallsWithType[U:! type](y: U) {
  // CHECK:STDERR: fail_todo_no_interfaces.carbon:[[@LINE+10]]:3: error: member access into undefined facet type `Core.ImplicitAs(type where...)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   TakesExtraWhereDeduced(y);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:13:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_no_interfaces.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TakesExtraWhereDeduced[T:! type where .Self impls type](x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakesExtraWhereDeduced(y);
}

fn TakesExtraWhereExplicit(T:! type where .Self impls type) {}
fn CallsWithTypeExplicit(U:! type) {
  // CHECK:STDERR: fail_todo_no_interfaces.carbon:[[@LINE+10]]:3: error: member access into undefined facet type `Core.ImplicitAs(type where...)` [QualifiedExprInUndefinedFacetTypeScope]
  // CHECK:STDERR:   TakesExtraWhereExplicit(U);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: include_files/facet_types.carbon:13:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface ImplicitAs(Dest:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_no_interfaces.carbon:[[@LINE-8]]:28: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn TakesExtraWhereExplicit(T:! type where .Self impls type) {}
  // CHECK:STDERR:                            ^
  // CHECK:STDERR:
  TakesExtraWhereExplicit(U);
}

// CHECK:STDOUT: --- convert_to_narrowed_facet_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %Eats.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %Eats.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Feed.type: type = fn_type @Feed [concrete]
// CHECK:STDOUT:   %Feed: %Feed.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %U.patt: <error> = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.3bf = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Feed = %Feed.decl
// CHECK:STDOUT:     .HandleAnimal = %HandleAnimal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Feed.decl: %Feed.type = fn_decl @Feed [concrete = constants.%Feed] {
// CHECK:STDOUT:     %T.patt.loc6_9.1: %Eats.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %e.patt: @Feed.%T.as_type.loc6_22.2 (%T.as_type) = binding_pattern e
// CHECK:STDOUT:     %e.param_patt: @Feed.%T.as_type.loc6_22.2 (%T.as_type) = value_param_pattern %e.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %T.loc6_9.1: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:     %e.param: @Feed.%T.as_type.loc6_22.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_22.1: type = splice_block %.loc6_22.2 [symbolic = %T.as_type.loc6_22.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Eats.type = name_ref T, %T.loc6_9.1 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6_22.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_22.2: type = converted %T.ref, %T.as_type.loc6_22.1 [symbolic = %T.as_type.loc6_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %e: @Feed.%T.as_type.loc6_22.2 (%T.as_type) = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %U.patt.loc8_17.1: <error> = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_17.2 (constants.%U.patt)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: <error> = bind_symbolic_name U, 0 [concrete = <error>]
// CHECK:STDOUT:     %a.param: <error> = value_param call_param0
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, %U [concrete = <error>]
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1b5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.fd4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.3bf
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Feed(%T.loc6_9.1: %Eats.type) {
// CHECK:STDOUT:   %T.loc6_9.2: %Eats.type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_9.2: %Eats.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc6_22.2: type = facet_access_type %T.loc6_9.2 [symbolic = %T.as_type.loc6_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Feed.%T.as_type.loc6_22.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_9.1: %Eats.type](%e.param_patt: @Feed.%T.as_type.loc6_22.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%U: <error>) {
// CHECK:STDOUT:   %U.patt.loc8_17.2: <error> = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_17.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%U.patt.loc8_17.1: <error>](%a.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Feed.ref: %Feed.type = name_ref Feed, file.%Feed.decl [concrete = constants.%Feed]
// CHECK:STDOUT:     %a.ref: <error> = name_ref a, %a [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Feed(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_9.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc6_22.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(<error>) {
// CHECK:STDOUT:   %U.patt.loc8_17.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bigger.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Tame.type: type = facet_type <@Tame> [concrete]
// CHECK:STDOUT:   %Self.7ee: %Tame.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: <error> = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %FeedTame.type: type = fn_type @FeedTame [concrete]
// CHECK:STDOUT:   %FeedTame: %FeedTame.type = struct_value () [concrete]
// CHECK:STDOUT:   %W.patt: <error> = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %HandleTameAnimal.type: type = fn_type @HandleTameAnimal [concrete]
// CHECK:STDOUT:   %HandleTameAnimal: %HandleTameAnimal.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.3bf = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Tame = %Tame.decl
// CHECK:STDOUT:     .FeedTame = %FeedTame.decl
// CHECK:STDOUT:     .HandleTameAnimal = %HandleTameAnimal.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Tame.decl: type = interface_decl @Tame [concrete = constants.%Tame.type] {} {}
// CHECK:STDOUT:   %FeedTame.decl: %FeedTame.type = fn_decl @FeedTame [concrete = constants.%FeedTame] {
// CHECK:STDOUT:     %V.patt.loc7_13.1: <error> = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc7_13.2 (constants.%V.patt)]
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: <error> = value_param_pattern %v.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V: <error> = bind_symbolic_name V, 0 [concrete = <error>]
// CHECK:STDOUT:     %v.param: <error> = value_param call_param0
// CHECK:STDOUT:     %V.ref: <error> = name_ref V, %V [concrete = <error>]
// CHECK:STDOUT:     %v: <error> = bind_name v, %v.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleTameAnimal.decl: %HandleTameAnimal.type = fn_decl @HandleTameAnimal [concrete = constants.%HandleTameAnimal] {
// CHECK:STDOUT:     %W.patt.loc9_21.1: <error> = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc9_21.2 (constants.%W.patt)]
// CHECK:STDOUT:     %w.patt: <error> = binding_pattern w
// CHECK:STDOUT:     %w.param_patt: <error> = value_param_pattern %w.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W: <error> = bind_symbolic_name W, 0 [concrete = <error>]
// CHECK:STDOUT:     %w.param: <error> = value_param call_param0
// CHECK:STDOUT:     %W.ref: <error> = name_ref W, %W [concrete = <error>]
// CHECK:STDOUT:     %w: <error> = bind_name w, %w.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1b5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.fd4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Tame {
// CHECK:STDOUT:   %Self: %Tame.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.3bf
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FeedTame(%V: <error>) {
// CHECK:STDOUT:   %V.patt.loc7_13.2: <error> = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc7_13.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%V.patt.loc7_13.1: <error>](%v.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleTameAnimal(%W: <error>) {
// CHECK:STDOUT:   %W.patt.loc9_21.2: <error> = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc9_21.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.patt.loc9_21.1: <error>](%w.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FeedTame.ref: %FeedTame.type = name_ref FeedTame, file.%FeedTame.decl [concrete = constants.%FeedTame]
// CHECK:STDOUT:     %w.ref: <error> = name_ref w, %w [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame(<error>) {
// CHECK:STDOUT:   %V.patt.loc7_13.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleTameAnimal(<error>) {
// CHECK:STDOUT:   %W.patt.loc9_21.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- with_blanket.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Eats.type: type = facet_type <@Eats> [concrete]
// CHECK:STDOUT:   %Self.1b5: %Eats.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Tame.type: type = facet_type <@Tame> [concrete]
// CHECK:STDOUT:   %Self.7ee: %Tame.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A: %Animal.type = bind_symbolic_name A, 0 [symbolic]
// CHECK:STDOUT:   %A.patt: %Animal.type = symbolic_binding_pattern A, 0 [symbolic]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%A) [symbolic]
// CHECK:STDOUT:   %V.patt: <error> = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %FeedTame2.type: type = fn_type @FeedTame2 [concrete]
// CHECK:STDOUT:   %FeedTame2: %FeedTame2.type = struct_value () [concrete]
// CHECK:STDOUT:   %W.patt: <error> = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %HandleTameAnimal2.type: type = fn_type @HandleTameAnimal2 [concrete]
// CHECK:STDOUT:   %HandleTameAnimal2: %HandleTameAnimal2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst101 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.3bf = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Eats = %Eats.decl
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .Tame = %Tame.decl
// CHECK:STDOUT:     .FeedTame2 = %FeedTame2.decl
// CHECK:STDOUT:     .HandleTameAnimal2 = %HandleTameAnimal2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eats.decl: type = interface_decl @Eats [concrete = constants.%Eats.type] {} {}
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %Tame.decl: type = interface_decl @Tame [concrete = constants.%Tame.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %A.patt.loc7_14.1: %Animal.type = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc7_14.2 (constants.%A.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %Animal.type = name_ref A, %A.loc7_14.1 [symbolic = %A.loc7_14.2 (constants.%A)]
// CHECK:STDOUT:     %A.as_type.loc7_26.1: type = facet_access_type %A.ref [symbolic = %A.as_type.loc7_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:     %.loc7: type = converted %A.ref, %A.as_type.loc7_26.1 [symbolic = %A.as_type.loc7_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:     %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %A.loc7_14.1: %Animal.type = bind_symbolic_name A, 0 [symbolic = %A.loc7_14.2 (constants.%A)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(constants.%A) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:   %FeedTame2.decl: %FeedTame2.type = fn_decl @FeedTame2 [concrete = constants.%FeedTame2] {
// CHECK:STDOUT:     %V.patt.loc9_14.1: <error> = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc9_14.2 (constants.%V.patt)]
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: <error> = value_param_pattern %v.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:       %Eats.ref: type = name_ref Eats, file.%Eats.decl [concrete = constants.%Eats.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V: <error> = bind_symbolic_name V, 0 [concrete = <error>]
// CHECK:STDOUT:     %v.param: <error> = value_param call_param0
// CHECK:STDOUT:     %V.ref: <error> = name_ref V, %V [concrete = <error>]
// CHECK:STDOUT:     %v: <error> = bind_name v, %v.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleTameAnimal2.decl: %HandleTameAnimal2.type = fn_decl @HandleTameAnimal2 [concrete = constants.%HandleTameAnimal2] {
// CHECK:STDOUT:     %W.patt.loc11_22.1: <error> = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_22.2 (constants.%W.patt)]
// CHECK:STDOUT:     %w.patt: <error> = binding_pattern w
// CHECK:STDOUT:     %w.param_patt: <error> = value_param_pattern %w.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:       %Tame.ref: type = name_ref Tame, file.%Tame.decl [concrete = constants.%Tame.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W: <error> = bind_symbolic_name W, 0 [concrete = <error>]
// CHECK:STDOUT:     %w.param: <error> = value_param call_param0
// CHECK:STDOUT:     %W.ref: <error> = name_ref W, %W [concrete = <error>]
// CHECK:STDOUT:     %w: <error> = bind_name w, %w.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eats {
// CHECK:STDOUT:   %Self: %Eats.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1b5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.fd4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Tame {
// CHECK:STDOUT:   %Self: %Tame.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.3bf
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%A.loc7_14.1: %Animal.type) {
// CHECK:STDOUT:   %A.loc7_14.2: %Animal.type = bind_symbolic_name A, 0 [symbolic = %A.loc7_14.2 (constants.%A)]
// CHECK:STDOUT:   %A.patt.loc7_14.2: %Animal.type = symbolic_binding_pattern A, 0 [symbolic = %A.patt.loc7_14.2 (constants.%A.patt)]
// CHECK:STDOUT:   %A.as_type.loc7_26.2: type = facet_access_type %A.loc7_14.2 [symbolic = %A.as_type.loc7_26.2 (constants.%A.as_type)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl(%A.loc7_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc7 as %Eats.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FeedTame2(%V: <error>) {
// CHECK:STDOUT:   %V.patt.loc9_14.2: <error> = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc9_14.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%V.patt.loc9_14.1: <error>](%v.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleTameAnimal2(%W: <error>) {
// CHECK:STDOUT:   %W.patt.loc11_22.2: <error> = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_22.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.patt.loc11_22.1: <error>](%w.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FeedTame2.ref: %FeedTame2.type = name_ref FeedTame2, file.%FeedTame2.decl [concrete = constants.%FeedTame2]
// CHECK:STDOUT:     %w.ref: <error> = name_ref w, %w [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%A) {
// CHECK:STDOUT:   %A.loc7_14.2 => constants.%A
// CHECK:STDOUT:   %A.patt.loc7_14.2 => constants.%A
// CHECK:STDOUT:   %A.as_type.loc7_26.2 => constants.%A.as_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%A.loc7_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedTame2(<error>) {
// CHECK:STDOUT:   %V.patt.loc9_14.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleTameAnimal2(<error>) {
// CHECK:STDOUT:   %W.patt.loc11_22.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- equivalent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %A.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.cf4: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A where TODO> [concrete]
// CHECK:STDOUT:   %U: %A_where.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %A_where.type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %WithExtraWhere.type: type = fn_type @WithExtraWhere [concrete]
// CHECK:STDOUT:   %WithExtraWhere: %WithExtraWhere.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.732: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %U.as_wit.iface0: <witness> = facet_access_witness %U, element0 [symbolic]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %U.as_type, (%U.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %TakesA.specific_fn: <specific function> = specific_function %TakesA, @TakesA(%A.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .WithExtraWhere = %WithExtraWhere.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt.loc7_11.1: %A.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_11.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @TakesA.%T.as_type.loc7_21.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @TakesA.%T.as_type.loc7_21.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %T.loc7_11.1: %A.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_11.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.as_type.loc7_21.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_21.1: type = splice_block %.loc7_21.2 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_11.1 [symbolic = %T.loc7_11.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_21.2: type = converted %T.ref, %T.as_type.loc7_21.1 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.as_type.loc7_21.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithExtraWhere.decl: %WithExtraWhere.type = fn_decl @WithExtraWhere [concrete = constants.%WithExtraWhere] {
// CHECK:STDOUT:     %U.patt.loc9_19.1: %A_where.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_19.2 (constants.%U.patt)]
// CHECK:STDOUT:     %y.patt: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) = value_param_pattern %y.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_25.1: type = splice_block %.loc9_25.2 [concrete = constants.%A_where.type] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %.Self: %A.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %A.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_31: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc9_25.2: type = where_expr %.Self [concrete = constants.%A_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc9_31, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc9_19.1: %A_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc9_19.2 (constants.%U)]
// CHECK:STDOUT:     %y.param: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_52.1: type = splice_block %.loc9_52.2 [symbolic = %U.as_type.loc9_52.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %A_where.type = name_ref U, %U.loc9_19.1 [symbolic = %U.loc9_19.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc9_52.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc9_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc9_52.2: type = converted %U.ref, %U.as_type.loc9_52.1 [symbolic = %U.as_type.loc9_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc7_11.1: %A.type) {
// CHECK:STDOUT:   %T.loc7_11.2: %A.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_11.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_11.2: %A.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_11.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc7_21.2: type = facet_access_type %T.loc7_11.2 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @TakesA.%T.as_type.loc7_21.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.cf4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc7_11.1: %A.type](%x.param_patt: @TakesA.%T.as_type.loc7_21.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WithExtraWhere(%U.loc9_19.1: %A_where.type) {
// CHECK:STDOUT:   %U.loc9_19.2: %A_where.type = bind_symbolic_name U, 0 [symbolic = %U.loc9_19.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc9_19.2: %A_where.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_19.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.as_type.loc9_52.2: type = facet_access_type %U.loc9_19.2 [symbolic = %U.as_type.loc9_52.2 (constants.%U.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) [symbolic = %require_complete (constants.%require_complete.732)]
// CHECK:STDOUT:   %U.as_wit.iface0.loc10_11.3: <witness> = facet_access_witness %U.loc9_19.2, element0 [symbolic = %U.as_wit.iface0.loc10_11.3 (constants.%U.as_wit.iface0)]
// CHECK:STDOUT:   %A.facet.loc10_11.3: %A.type = facet_value %U.as_type.loc9_52.2, (%U.as_wit.iface0.loc10_11.3) [symbolic = %A.facet.loc10_11.3 (constants.%A.facet)]
// CHECK:STDOUT:   %TakesA.specific_fn.loc10_3.2: <specific function> = specific_function constants.%TakesA, @TakesA(%A.facet.loc10_11.3) [symbolic = %TakesA.specific_fn.loc10_3.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%U.patt.loc9_19.1: %A_where.type](%y.param_patt: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %y.ref: @WithExtraWhere.%U.as_type.loc9_52.2 (%U.as_type) = name_ref y, %y
// CHECK:STDOUT:     %U.as_wit.iface0.loc10_11.1: <witness> = facet_access_witness constants.%U, element0 [symbolic = %U.as_wit.iface0.loc10_11.3 (constants.%U.as_wit.iface0)]
// CHECK:STDOUT:     %A.facet.loc10_11.1: %A.type = facet_value constants.%U.as_type, (%U.as_wit.iface0.loc10_11.1) [symbolic = %A.facet.loc10_11.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc10_11.1: %A.type = converted constants.%U.as_type, %A.facet.loc10_11.1 [symbolic = %A.facet.loc10_11.3 (constants.%A.facet)]
// CHECK:STDOUT:     %U.as_wit.iface0.loc10_11.2: <witness> = facet_access_witness constants.%U, element0 [symbolic = %U.as_wit.iface0.loc10_11.3 (constants.%U.as_wit.iface0)]
// CHECK:STDOUT:     %A.facet.loc10_11.2: %A.type = facet_value constants.%U.as_type, (%U.as_wit.iface0.loc10_11.2) [symbolic = %A.facet.loc10_11.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc10_11.2: %A.type = converted constants.%U.as_type, %A.facet.loc10_11.2 [symbolic = %A.facet.loc10_11.3 (constants.%A.facet)]
// CHECK:STDOUT:     %TakesA.specific_fn.loc10_3.1: <specific function> = specific_function %TakesA.ref, @TakesA(constants.%A.facet) [symbolic = %TakesA.specific_fn.loc10_3.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:     %TakesA.call: init %empty_tuple.type = call %TakesA.specific_fn.loc10_3.1(%y.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.loc7_11.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_11.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_21.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithExtraWhere(constants.%U) {
// CHECK:STDOUT:   %U.loc9_19.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc9_19.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc9_52.2 => constants.%U.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc7_11.2 => constants.%A.facet
// CHECK:STDOUT:   %T.patt.loc7_11.2 => constants.%A.facet
// CHECK:STDOUT:   %T.as_type.loc7_21.2 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.732
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(@WithExtraWhere.%A.facet.loc10_11.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- no_interfaces_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %TakesTypeDeduced.type: type = fn_type @TakesTypeDeduced [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesTypeDeduced: %TakesTypeDeduced.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %U: %type_where = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %type_where = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %CallsWithExtraWhere.type: type = fn_type @CallsWithExtraWhere [concrete]
// CHECK:STDOUT:   %CallsWithExtraWhere: %CallsWithExtraWhere.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.220: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %TakesTypeDeduced.specific_fn: <specific function> = specific_function %TakesTypeDeduced, @TakesTypeDeduced(%U.as_type) [symbolic]
// CHECK:STDOUT:   %TakesTypeExplicit.type: type = fn_type @TakesTypeExplicit [concrete]
// CHECK:STDOUT:   %TakesTypeExplicit: %TakesTypeExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallsWithExtraWhereExplicit.type: type = fn_type @CallsWithExtraWhereExplicit [concrete]
// CHECK:STDOUT:   %CallsWithExtraWhereExplicit: %CallsWithExtraWhereExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakesTypeExplicit.specific_fn: <specific function> = specific_function %TakesTypeExplicit, @TakesTypeExplicit(%U.as_type) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TakesTypeDeduced = %TakesTypeDeduced.decl
// CHECK:STDOUT:     .CallsWithExtraWhere = %CallsWithExtraWhere.decl
// CHECK:STDOUT:     .TakesTypeExplicit = %TakesTypeExplicit.decl
// CHECK:STDOUT:     .CallsWithExtraWhereExplicit = %CallsWithExtraWhereExplicit.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TakesTypeDeduced.decl: %TakesTypeDeduced.type = fn_decl @TakesTypeDeduced [concrete = constants.%TakesTypeDeduced] {
// CHECK:STDOUT:     %T.patt.loc3_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_21.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @TakesTypeDeduced.%T.loc3_21.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @TakesTypeDeduced.%T.loc3_21.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_21.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesTypeDeduced.%T.loc3_21.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc3_21.1 [symbolic = %T.loc3_21.2 (constants.%T)]
// CHECK:STDOUT:     %x: @TakesTypeDeduced.%T.loc3_21.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallsWithExtraWhere.decl: %CallsWithExtraWhere.type = fn_decl @CallsWithExtraWhere [concrete = constants.%CallsWithExtraWhere] {
// CHECK:STDOUT:     %U.patt.loc4_24.1: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc4_24.2 (constants.%U.patt)]
// CHECK:STDOUT:     %y.patt: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) = value_param_pattern %y.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_33.1: type = splice_block %.loc4_33.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_33.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc4_24.1: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc4_24.2 (constants.%U)]
// CHECK:STDOUT:     %y.param: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_60.1: type = splice_block %.loc4_60.2 [symbolic = %U.as_type.loc4_60.2 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %type_where = name_ref U, %U.loc4_24.1 [symbolic = %U.loc4_24.2 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc4_60.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc4_60.2 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc4_60.2: type = converted %U.ref, %U.as_type.loc4_60.1 [symbolic = %U.as_type.loc4_60.2 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakesTypeExplicit.decl: %TakesTypeExplicit.type = fn_decl @TakesTypeExplicit [concrete = constants.%TakesTypeExplicit] {
// CHECK:STDOUT:     %T.patt.loc8_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallsWithExtraWhereExplicit.decl: %CallsWithExtraWhereExplicit.type = fn_decl @CallsWithExtraWhereExplicit [concrete = constants.%CallsWithExtraWhereExplicit] {
// CHECK:STDOUT:     %U.patt.loc9_32.1: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_32.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_41.1: type = splice_block %.loc9_41.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9_41.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc9_32.1: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc9_32.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesTypeDeduced(%T.loc3_21.1: type) {
// CHECK:STDOUT:   %T.loc3_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc3_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_21.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @TakesTypeDeduced.%T.loc3_21.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc3_21.1: type](%x.param_patt: @TakesTypeDeduced.%T.loc3_21.2 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallsWithExtraWhere(%U.loc4_24.1: %type_where) {
// CHECK:STDOUT:   %U.loc4_24.2: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc4_24.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc4_24.2: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc4_24.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.as_type.loc4_60.2: type = facet_access_type %U.loc4_24.2 [symbolic = %U.as_type.loc4_60.2 (constants.%U.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) [symbolic = %require_complete (constants.%require_complete.220)]
// CHECK:STDOUT:   %TakesTypeDeduced.specific_fn.loc5_3.2: <specific function> = specific_function constants.%TakesTypeDeduced, @TakesTypeDeduced(%U.as_type.loc4_60.2) [symbolic = %TakesTypeDeduced.specific_fn.loc5_3.2 (constants.%TakesTypeDeduced.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%U.patt.loc4_24.1: %type_where](%y.param_patt: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesTypeDeduced.ref: %TakesTypeDeduced.type = name_ref TakesTypeDeduced, file.%TakesTypeDeduced.decl [concrete = constants.%TakesTypeDeduced]
// CHECK:STDOUT:     %y.ref: @CallsWithExtraWhere.%U.as_type.loc4_60.2 (%U.as_type) = name_ref y, %y
// CHECK:STDOUT:     %TakesTypeDeduced.specific_fn.loc5_3.1: <specific function> = specific_function %TakesTypeDeduced.ref, @TakesTypeDeduced(constants.%U.as_type) [symbolic = %TakesTypeDeduced.specific_fn.loc5_3.2 (constants.%TakesTypeDeduced.specific_fn)]
// CHECK:STDOUT:     %TakesTypeDeduced.call: init %empty_tuple.type = call %TakesTypeDeduced.specific_fn.loc5_3.1(%y.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesTypeExplicit(%T.loc8_22.1: type) {
// CHECK:STDOUT:   %T.loc8_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_22.1: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallsWithExtraWhereExplicit(%U.loc9_32.1: %type_where) {
// CHECK:STDOUT:   %U.loc9_32.2: %type_where = bind_symbolic_name U, 0 [symbolic = %U.loc9_32.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc9_32.2: %type_where = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc9_32.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.as_type.loc10_22.2: type = facet_access_type %U.loc9_32.2 [symbolic = %U.as_type.loc10_22.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %TakesTypeExplicit.specific_fn.loc10_3.2: <specific function> = specific_function constants.%TakesTypeExplicit, @TakesTypeExplicit(%U.as_type.loc10_22.2) [symbolic = %TakesTypeExplicit.specific_fn.loc10_3.2 (constants.%TakesTypeExplicit.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc9_32.1: %type_where) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesTypeExplicit.ref: %TakesTypeExplicit.type = name_ref TakesTypeExplicit, file.%TakesTypeExplicit.decl [concrete = constants.%TakesTypeExplicit]
// CHECK:STDOUT:     %U.ref: %type_where = name_ref U, %U.loc9_32.1 [symbolic = %U.loc9_32.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc10_22.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc10_22.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc10: type = converted %U.ref, %U.as_type.loc10_22.1 [symbolic = %U.as_type.loc10_22.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %TakesTypeExplicit.specific_fn.loc10_3.1: <specific function> = specific_function %TakesTypeExplicit.ref, @TakesTypeExplicit(constants.%U.as_type) [symbolic = %TakesTypeExplicit.specific_fn.loc10_3.2 (constants.%TakesTypeExplicit.specific_fn)]
// CHECK:STDOUT:     %TakesTypeExplicit.call: init %empty_tuple.type = call %TakesTypeExplicit.specific_fn.loc10_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeDeduced(constants.%T) {
// CHECK:STDOUT:   %T.loc3_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc3_21.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallsWithExtraWhere(constants.%U) {
// CHECK:STDOUT:   %U.loc4_24.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc4_24.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc4_60.2 => constants.%U.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeDeduced(constants.%U.as_type) {
// CHECK:STDOUT:   %T.loc3_21.2 => constants.%U.as_type
// CHECK:STDOUT:   %T.patt.loc3_21.2 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeDeduced(@CallsWithExtraWhere.%U.as_type.loc4_60.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeExplicit(constants.%T) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallsWithExtraWhereExplicit(constants.%U) {
// CHECK:STDOUT:   %U.loc9_32.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc9_32.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeExplicit(constants.%U.as_type) {
// CHECK:STDOUT:   %T.loc8_22.2 => constants.%U.as_type
// CHECK:STDOUT:   %T.patt.loc8_22.2 => constants.%U.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesTypeExplicit(@CallsWithExtraWhereExplicit.%U.as_type.loc10_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_no_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [concrete]
// CHECK:STDOUT:   %T: %type_where = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %type_where = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %TakesExtraWhereDeduced.type: type = fn_type @TakesExtraWhereDeduced [concrete]
// CHECK:STDOUT:   %TakesExtraWhereDeduced: %TakesExtraWhereDeduced.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.220: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %CallsWithType.type: type = fn_type @CallsWithType [concrete]
// CHECK:STDOUT:   %CallsWithType: %CallsWithType.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert: %Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %TakesExtraWhereExplicit.type: type = fn_type @TakesExtraWhereExplicit [concrete]
// CHECK:STDOUT:   %TakesExtraWhereExplicit: %TakesExtraWhereExplicit.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallsWithTypeExplicit.type: type = fn_type @CallsWithTypeExplicit [concrete]
// CHECK:STDOUT:   %CallsWithTypeExplicit: %CallsWithTypeExplicit.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//prelude, inst66 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.31f = import_ref Core//prelude, loc14_35, unloaded
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//prelude, inst66 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//prelude, loc14_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TakesExtraWhereDeduced = %TakesExtraWhereDeduced.decl
// CHECK:STDOUT:     .CallsWithType = %CallsWithType.decl
// CHECK:STDOUT:     .TakesExtraWhereExplicit = %TakesExtraWhereExplicit.decl
// CHECK:STDOUT:     .CallsWithTypeExplicit = %CallsWithTypeExplicit.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TakesExtraWhereDeduced.decl: %TakesExtraWhereDeduced.type = fn_decl @TakesExtraWhereDeduced [concrete = constants.%TakesExtraWhereDeduced] {
// CHECK:STDOUT:     %T.patt.loc3_27.1: %type_where = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_27.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_36.1: type = splice_block %.loc3_36.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_36.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_27.1: %type_where = bind_symbolic_name T, 0 [symbolic = %T.loc3_27.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc3_63.1: type = splice_block %.loc3_63.2 [symbolic = %T.as_type.loc3_63.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %type_where = name_ref T, %T.loc3_27.1 [symbolic = %T.loc3_27.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc3_63.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc3_63.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc3_63.2: type = converted %T.ref, %T.as_type.loc3_63.1 [symbolic = %T.as_type.loc3_63.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallsWithType.decl: %CallsWithType.type = fn_decl @CallsWithType [concrete = constants.%CallsWithType] {
// CHECK:STDOUT:     %U.patt.loc4_18.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc4_18.2 (constants.%U.patt)]
// CHECK:STDOUT:     %y.patt: @CallsWithType.%U.loc4_18.2 (%U) = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: @CallsWithType.%U.loc4_18.2 (%U) = value_param_pattern %y.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc4_18.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_18.2 (constants.%U)]
// CHECK:STDOUT:     %y.param: @CallsWithType.%U.loc4_18.2 (%U) = value_param call_param0
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc4_18.1 [symbolic = %U.loc4_18.2 (constants.%U)]
// CHECK:STDOUT:     %y: @CallsWithType.%U.loc4_18.2 (%U) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakesExtraWhereExplicit.decl: %TakesExtraWhereExplicit.type = fn_decl @TakesExtraWhereExplicit [concrete = constants.%TakesExtraWhereExplicit] {
// CHECK:STDOUT:     %T.patt.loc18_28.1: %type_where = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_28.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_37.1: type = splice_block %.loc18_37.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc18_37.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_28.1: %type_where = bind_symbolic_name T, 0 [symbolic = %T.loc18_28.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallsWithTypeExplicit.decl: %CallsWithTypeExplicit.type = fn_decl @CallsWithTypeExplicit [concrete = constants.%CallsWithTypeExplicit] {
// CHECK:STDOUT:     %U.patt.loc19_26.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc19_26.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc19_26.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc19_26.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type)]
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
// CHECK:STDOUT: generic fn @TakesExtraWhereDeduced(%T.loc3_27.1: %type_where) {
// CHECK:STDOUT:   %T.loc3_27.2: %type_where = bind_symbolic_name T, 0 [symbolic = %T.loc3_27.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc3_27.2: %type_where = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_27.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc3_63.2: type = facet_access_type %T.loc3_27.2 [symbolic = %T.as_type.loc3_63.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.220)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc3_27.1: %type_where](%x.param_patt: @TakesExtraWhereDeduced.%T.as_type.loc3_63.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallsWithType(%U.loc4_18.1: type) {
// CHECK:STDOUT:   %U.loc4_18.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_18.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc4_18.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc4_18.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @CallsWithType.%U.loc4_18.2 (%U) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%U.patt.loc4_18.1: type](%y.param_patt: @CallsWithType.%U.loc4_18.2 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesExtraWhereDeduced.ref: %TakesExtraWhereDeduced.type = name_ref TakesExtraWhereDeduced, file.%TakesExtraWhereDeduced.decl [concrete = constants.%TakesExtraWhereDeduced]
// CHECK:STDOUT:     %y.ref: @CallsWithType.%U.loc4_18.2 (%U) = name_ref y, %y
// CHECK:STDOUT:     %.loc15: %type_where = converted constants.%U, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesExtraWhereExplicit(%T.loc18_28.1: %type_where) {
// CHECK:STDOUT:   %T.loc18_28.2: %type_where = bind_symbolic_name T, 0 [symbolic = %T.loc18_28.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc18_28.2: %type_where = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_28.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc18_28.1: %type_where) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallsWithTypeExplicit(%U.loc19_26.1: type) {
// CHECK:STDOUT:   %U.loc19_26.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc19_26.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc19_26.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc19_26.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc19_26.1: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesExtraWhereExplicit.ref: %TakesExtraWhereExplicit.type = name_ref TakesExtraWhereExplicit, file.%TakesExtraWhereExplicit.decl [concrete = constants.%TakesExtraWhereExplicit]
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc19_26.1 [symbolic = %U.loc19_26.2 (constants.%U)]
// CHECK:STDOUT:     %.loc30: %type_where = converted %U.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesExtraWhereDeduced(constants.%T) {
// CHECK:STDOUT:   %T.loc3_27.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc3_27.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc3_63.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallsWithType(constants.%U) {
// CHECK:STDOUT:   %U.loc4_18.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc4_18.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%type_where) {
// CHECK:STDOUT:   %Dest => constants.%type_where
// CHECK:STDOUT:   %Dest.patt => constants.%type_where
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesExtraWhereExplicit(constants.%T) {
// CHECK:STDOUT:   %T.loc18_28.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc18_28.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallsWithTypeExplicit(constants.%U) {
// CHECK:STDOUT:   %U.loc19_26.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc19_26.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.121: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc9_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc13_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc21_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc9_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc9_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc9_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc10_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc9_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc10_20.1: type = splice_block %.loc10_20.3 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc10_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc10_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc10_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc10_20.3: type = converted %Self.ref, %Self.as_type.loc10_20.2 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc13_22.1: type) {
// CHECK:STDOUT:   %Dest.loc13_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc13_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc13_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc18_37: type = facet_access_type %Self.ref.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc18_37: type = converted %Self.ref.loc18_37, %Self.as_type.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref.loc18_15 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref.loc18_15, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_28: type = facet_access_type %Self.ref.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_28.2: type = converted %Self.ref.loc18_28, %Self.as_type.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.121]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc21_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%T.loc21_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T.loc21_14.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%T (%T) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%T (%T) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%T (%T) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc22_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc22_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc22_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)](%other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc9_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc13_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc21_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
