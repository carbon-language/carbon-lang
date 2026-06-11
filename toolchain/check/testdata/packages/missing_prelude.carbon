// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/missing_prelude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/missing_prelude.carbon

// --- fail_missing_prelude.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_missing_prelude.carbon:[[@LINE+4]]:8: error: `Core.Int` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: var n: i32;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
var n: i32;

// --- prelude_empty.carbon

package Core library "[[@TEST_NAME]]";

// --- fail_missing_prelude_member.carbon

library "[[@TEST_NAME]]";

import Core library "prelude_empty";

// CHECK:STDERR: fail_missing_prelude_member.carbon:[[@LINE+4]]:8: error: name `Core.Int` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: var n: i32;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
var n: i32;

// --- prelude_fake_int.carbon

package Core library "[[@TEST_NAME]]";

class X {}

fn Int[T:! type](unused N:! T) -> X { return {}; }

interface ImplicitAs(T:! type) {
  fn Convert(self) -> T;
}

// --- fail_use_fake_int.carbon

library "[[@TEST_NAME]]";

import Core library "prelude_fake_int";

// We don't really care whether this works, but it shouldn't crash or otherwise misbehave.
// CHECK:STDERR: fail_use_fake_int.carbon:[[@LINE+14]]:17: error: cannot implicitly convert non-type value of type `Core.X` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: let n: Core.X = i32;
// CHECK:STDERR:                 ^~~
// CHECK:STDERR: fail_use_fake_int.carbon:[[@LINE+11]]:17: note: type `Core.X` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let n: Core.X = i32;
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
// CHECK:STDERR: fail_use_fake_int.carbon:[[@LINE+7]]:17: error: cannot implicitly convert expression of type `type` to `Core.X` [ConversionFailure]
// CHECK:STDERR: let n: Core.X = i32;
// CHECK:STDERR:                 ^~~
// CHECK:STDERR: fail_use_fake_int.carbon:[[@LINE+4]]:17: note: type `type` does not implement interface `Core.ImplicitAs(Core.X)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let n: Core.X = i32;
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
let n: Core.X = i32;

// --- prelude_use_in_prelude.carbon

package Core library "[[@TEST_NAME]]";

// Core is not an imported package here.

class Int[T:! type](unused N:! T) {}

let n: i32 = {};

// --- fail_prelude_as_namespace.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_prelude_as_namespace.carbon:[[@LINE+8]]:11: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: namespace Core;
// CHECK:STDERR:           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_prelude_as_namespace.carbon:[[@LINE+4]]:11: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: namespace Core;
// CHECK:STDERR:           ^~~~
// CHECK:STDERR:
namespace Core;

// CHECK:STDERR: fail_prelude_as_namespace.carbon:[[@LINE+4]]:4: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn Core.Int[T:! type](N:! T) -> {} { return {}; }
// CHECK:STDERR:    ^~~~
// CHECK:STDERR:
fn Core.Int[T:! type](N:! T) -> {} { return {}; }

// Make sure use of `i32` doesn't crash after above errors.
let n: type = i32;

// --- fail_prelude_as_class.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_prelude_as_class.carbon:[[@LINE+8]]:7: error: `class` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: class Core {
// CHECK:STDERR:       ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_prelude_as_class.carbon:[[@LINE+4]]:7: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: class Core {
// CHECK:STDERR:       ^~~~
// CHECK:STDERR:
class Core {
  fn Int[T:! type](N:! T) -> {} { return {}; }
}

let n: type = i32;


// CHECK:STDOUT: --- fail_missing_prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: <error> = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: <error> = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref <error> = var %n.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc8: type = type_literal <error> [concrete = <error>]
// CHECK:STDOUT:   %n: ref <error> = ref_binding n, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   assign file.%n.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_prelude_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     import Core//prelude_empty
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: <error> = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: <error> = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref <error> = var %n.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc10: type = type_literal <error> [concrete = <error>]
// CHECK:STDOUT:   %n: ref <error> = ref_binding n, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   assign file.%n.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_fake_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %N: %T = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %.7b0: Core.Form = init_form %X [concrete]
// CHECK:STDOUT:   %pattern_type.e00: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.701: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T, %Self) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert: %ImplicitAs.WithSelf.Convert.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: @Int.%pattern_type (%pattern_type.51d) = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e00 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.e00 = return_slot_pattern %return.param_patt, %X.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc6_35: Core.Form = init_form %X.ref [concrete = constants.%.7b0]
// CHECK:STDOUT:     %.loc6_12.1: type = splice_block %.loc6_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc6_9: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_29: type = splice_block %T.ref [symbolic = %T.loc6_9.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.loc6_26: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_9.2 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_26.2: @Int.%T.loc6_9.1 (%T) = symbolic_binding N, 1 [symbolic = %N.loc6_26.1 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.0ff = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_26.1: type = splice_block %.loc8_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_23.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc8_23.2: type) {
// CHECK:STDOUT:   %T.loc8_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_23.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc8_23.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self.loc8_32.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self.loc8_32.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_32.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self.loc8_32.2 (constants.%Self)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert)] {
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc9_14 (%pattern_type.701) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc9_14 (%pattern_type.701) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc9_23 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc9_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc8_23.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc9_23.2: Core.Form = init_form %T.ref [symbolic = %.loc9_23.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc9_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_14.1: type = splice_block %.loc9_14.3 [symbolic = %Self.as_type.loc9_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc9_14.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = specific_constant @ImplicitAs.%Self.loc8_32.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = name_ref Self, %.loc9_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc9_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc9_14.3: type = converted %Self.ref, %Self.as_type.loc9_14.2 [symbolic = %Self.as_type.loc9_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc9_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.WithSelf.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.WithSelf.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc9_24.1: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %ImplicitAs.WithSelf.Convert.decl [symbolic = %assoc0.loc9_24.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_32.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = @ImplicitAs.WithSelf.%assoc0.loc9_24.1
// CHECK:STDOUT:     witness = (@ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Int(%T.loc6_9.2: type, %N.loc6_26.2: @Int.%T.loc6_9.1 (%T)) {
// CHECK:STDOUT:   %T.loc6_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc6_26.1: @Int.%T.loc6_9.1 (%T) = symbolic_binding N, 1 [symbolic = %N.loc6_26.1 (constants.%N)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_9.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %X {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc6_47.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_47.2: init %X to %return.param = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:     %.loc6_48: init %X = converted %.loc6_47.1, %.loc6_47.2 [concrete = constants.%X.val]
// CHECK:STDOUT:     return %.loc6_48 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%T.loc8_23.2: type, @ImplicitAs.%Self.loc8_32.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_14: type = pattern_type %Self.as_type.loc9_14.1 [symbolic = %pattern_type.loc9_14 (constants.%pattern_type.701)]
// CHECK:STDOUT:   %.loc9_23.1: Core.Form = init_form %T [symbolic = %.loc9_23.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc9_23: type = pattern_type %T [symbolic = %pattern_type.loc9_23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc9_14.1 (%Self.as_type)) -> out %return.param: @ImplicitAs.WithSelf.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%T
// CHECK:STDOUT:   %N.loc6_26.1 => constants.%N
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc8_23.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3aa
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.701
// CHECK:STDOUT:   %.loc9_23.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc9_23 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_fake_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.e00: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %N: %T = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.specific_fn: <specific function> = specific_function %Int, @Int(Core.IntLiteral, %int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.294: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.fcc: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.945: %ImplicitAs.WithSelf.Convert.type.97e = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.294 [symbolic]
// CHECK:STDOUT:   %pattern_type.20e: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5f1: type = facet_type <@ImplicitAs, @ImplicitAs(type)> [concrete]
// CHECK:STDOUT:   %Self.966: %ImplicitAs.type.5f1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.082: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(type, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.016: %ImplicitAs.WithSelf.Convert.type.082 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.5b7: type = assoc_entity_type @ImplicitAs, @ImplicitAs(type) [concrete]
// CHECK:STDOUT:   %assoc0.08d: %ImplicitAs.assoc_type.5b7 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %assoc0.0ac: %ImplicitAs.assoc_type.fcc = assoc_entity element0, imports.%Core.import_ref.f9d [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.52b: type = facet_type <@ImplicitAs, @ImplicitAs(%X)> [concrete]
// CHECK:STDOUT:   %Self.ad7: %ImplicitAs.type.52b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.721: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%X, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.c03: %ImplicitAs.WithSelf.Convert.type.721 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.aff: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%X) [concrete]
// CHECK:STDOUT:   %assoc0.894: %ImplicitAs.assoc_type.aff = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .X = %Core.X
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude_fake_int
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.X: type = import_ref Core//prelude_fake_int, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Core.import_ref.8f2: <witness> = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Core.import_ref.eff = import_ref Core//prelude_fake_int, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude_fake_int, Int, loaded [concrete = constants.%Int]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.1: type = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, loaded [symbolic = @Int.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.b42: @Int.%T (%T) = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, loaded [symbolic = @Int.%N (constants.%N)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude_fake_int, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.484: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.fcc) = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.0ac)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude_fake_int, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.cb2: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.97e) = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.945)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.3: type = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.f19eaa.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Self (constants.%Self.294)]
// CHECK:STDOUT:   %Core.import_ref.8c8 = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.4: type = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.f9d = import_ref Core//prelude_fake_int, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.e00 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_12: type = splice_block %X.ref [concrete = constants.%X] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Core.X [concrete = constants.%X]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_17: %X = converted @__global_init.%.loc21_17.3, <error> [concrete = <error>]
// CHECK:STDOUT:   %n: %X = value_binding n, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.b3bc94.4: type) [from "prelude_fake_int.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.294)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.8c8
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.484
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "prelude_fake_int.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Core.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.eff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Int(imports.%Core.import_ref.b3bc94.1: type, imports.%Core.import_ref.b42: @Int.%T (%T)) [from "prelude_fake_int.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: @Int.%T (%T) = symbolic_binding N, 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(imports.%Core.import_ref.b3bc94.3: type, imports.%Core.import_ref.f19eaa.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa)) [from "prelude_fake_int.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.294)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.20e)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int.specific_fn: <specific function> = specific_function constants.%Int, @Int(Core.IntLiteral, constants.%int_32) [concrete = constants.%Int.specific_fn]
// CHECK:STDOUT:   %.loc21_17.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Int.call: init %X to %.loc21_17.1 = call %Int.specific_fn()
// CHECK:STDOUT:   %.loc21_17.2: type = converted %Int.call, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc21_17.3: type = type_literal <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(Core.IntLiteral, constants.%int_32) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dc0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%T, constants.%Self.294) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%T, constants.%Self.294) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3aa
// CHECK:STDOUT:   %Self => constants.%Self.294
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.20e
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5f1
// CHECK:STDOUT:   %Self => constants.%Self.966
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(type, constants.%Self.294) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5f1
// CHECK:STDOUT:   %Self => constants.%Self.294
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.082
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.016
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.5b7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.08d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.52b
// CHECK:STDOUT:   %Self => constants.%Self.ad7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%X, constants.%Self.294) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.52b
// CHECK:STDOUT:   %Self => constants.%Self.294
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.721
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.c03
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.aff
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.894
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- prelude_use_in_prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %N: %T = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.a4d: type = class_type @Int, @Int(%T, %N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.382: type = class_type @Int, @Int(Core.IntLiteral, %int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.0c6: type = pattern_type %Int.382 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.val: %Int.382 = struct_value () [concrete]
// CHECK:STDOUT:   %.d67: ref %Int.382 = temporary invalid, %Int.val [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = class_decl @Int [concrete = constants.%Int.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: @Int.%pattern_type (%pattern_type.51d) = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_15.1: type = splice_block %.loc6_15.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc6_12: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_15.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_12.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_12.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_32: type = splice_block %T.ref [symbolic = %T.loc6_12.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.loc6_29: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_12.2 [symbolic = %T.loc6_12.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_29.2: @Int.%T.loc6_12.1 (%T) = symbolic_binding N, 1 [symbolic = %N.loc6_29.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.0c6 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_8: type = type_literal constants.%Int.382 [concrete = constants.%Int.382]
// CHECK:STDOUT:   %.loc8_15.1: ref %Int.382 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.2: init %Int.382 to %.loc8_15.1 = class_init () [concrete = constants.%Int.val]
// CHECK:STDOUT:   %.loc8_15.3: init %Int.382 = converted @__global_init.%.loc8, %.loc8_15.2 [concrete = constants.%Int.val]
// CHECK:STDOUT:   %.loc8_15.4: ref %Int.382 = temporary %.loc8_15.1, %.loc8_15.3 [concrete = constants.%.d67]
// CHECK:STDOUT:   %.loc8_15.5: %Int.382 = acquire_value %.loc8_15.4 [concrete = constants.%Int.val]
// CHECK:STDOUT:   %n: %Int.382 = value_binding n, %.loc8_15.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Int(%T.loc6_12.2: type, %N.loc6_29.2: @Int.%T.loc6_12.1 (%T)) {
// CHECK:STDOUT:   %T.loc6_12.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_12.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc6_29.1: @Int.%T.loc6_12.1 (%T) = symbolic_binding N, 1 [symbolic = %N.loc6_29.1 (constants.%N)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_12.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Int.a4d
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6_12.1 => constants.%T
// CHECK:STDOUT:   %N.loc6_29.1 => constants.%N
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(Core.IntLiteral, constants.%int_32) {
// CHECK:STDOUT:   %T.loc6_12.1 => Core.IntLiteral
// CHECK:STDOUT:   %N.loc6_29.1 => constants.%int_32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dc0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_prelude_as_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_prelude_as_class.carbon
// CHECK:STDOUT:
