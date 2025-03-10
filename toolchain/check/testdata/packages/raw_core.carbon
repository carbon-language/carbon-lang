// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/raw_core.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/raw_core.carbon

// --- package_raw_core.carbon

package r#Core;

fn F();

// --- import_raw_core.carbon

library "[[@TEST_NAME]]";

import r#Core;

fn G() { r#Core.F(); }

fn H() { Core.Int(32); }

// --- fail_raw_core_not_core.carbon

library "[[@TEST_NAME]]";

import r#Core;

// CHECK:STDERR: fail_raw_core_not_core.carbon:[[@LINE+4]]:10: error: member name `F` not found in `Core` [MemberNameNotFoundInScope]
// CHECK:STDERR: fn G() { Core.F(); }
// CHECK:STDERR:          ^~~~~~
// CHECK:STDERR:
fn G() { Core.F(); }

// CHECK:STDERR: fail_raw_core_not_core.carbon:[[@LINE+4]]:10: error: member name `Int` not found in `r#Core` [MemberNameNotFoundInScope]
// CHECK:STDERR: fn H() { r#Core.Int(32); }
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn H() { r#Core.Int(32); }

// --- class_raw_core.carbon

class r#Core {
  var n: Core.Int(32);
}

var c: r#Core = {.n = 0 as Core.Int(32)};

// CHECK:STDOUT: --- package_raw_core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_raw_core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.eced1c.1: <namespace> = namespace file.%Core.import.1, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.eced1c.2: <namespace> = namespace file.%Core.import.loc4, [concrete] {
// CHECK:STDOUT:     .F = %Core.F
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.F: %F.type = import_ref Core//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core.eced1c.1
// CHECK:STDOUT:     .r#Core = imports.%Core.eced1c.2
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import.1 = import Core
// CHECK:STDOUT:   %Core.import.loc4 = import r#Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref r#Core, imports.%Core.eced1c.2 [concrete = imports.%Core.eced1c.2]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Core.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() [from "package_raw_core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core.eced1c.1 [concrete = imports.%Core.eced1c.1]
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_raw_core_not_core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.eced1c.1: <namespace> = namespace file.%Core.import.1, [concrete] {
// CHECK:STDOUT:     .F = <poisoned>
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.eced1c.2: <namespace> = namespace file.%Core.import.loc4, [concrete] {
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core.eced1c.1
// CHECK:STDOUT:     .r#Core = imports.%Core.eced1c.2
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import.1 = import Core
// CHECK:STDOUT:   %Core.import.loc4 = import r#Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core.eced1c.1 [concrete = imports.%Core.eced1c.1]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref r#Core, imports.%Core.eced1c.2 [concrete = imports.%Core.eced1c.2]
// CHECK:STDOUT:   %Int.ref: <error> = name_ref Int, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class_raw_core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Core: type = class_type @Core [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Core.elem: type = unbound_element_type %Core, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.882: <witness> = impl_witness (imports.%Core.import_ref.78a), @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%impl_witness.882) [concrete]
// CHECK:STDOUT:   %.249: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.val: %Core = struct_value (%int_0.6a9) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .r#Core = %Core.decl
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core.decl: type = class_decl @Core [concrete = constants.%Core] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %Core = binding_pattern c
// CHECK:STDOUT:     %.loc6: %Core = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Core = var c
// CHECK:STDOUT:   %Core.ref: type = name_ref r#Core, %Core.decl [concrete = constants.%Core]
// CHECK:STDOUT:   %c: ref %Core = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Core {
// CHECK:STDOUT:   %.loc3_8: %Core.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc3_3: %Core.elem = var_pattern %.loc3_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Core.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Core
// CHECK:STDOUT:   .n = %.loc3_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Int.ref: %Int.type = name_ref Int, imports.%Core.Int [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.249 = impl_witness_access constants.%impl_witness.882, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc6_25.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_25.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc6_25.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_25.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_25.2: %i32 = converted %int_0, %.loc6_25.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.1: %struct_type.n = struct_literal (%.loc6_25.2)
// CHECK:STDOUT:   %.loc6_40.2: ref %i32 = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_40.3: init %i32 = initialize_from %.loc6_25.2 to %.loc6_40.2 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.4: init %Core = class_init (%.loc6_40.3), file.%c.var [concrete = constants.%Core.val]
// CHECK:STDOUT:   %.loc6_1: init %Core = converted %.loc6_40.1, %.loc6_40.4 [concrete = constants.%Core.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
