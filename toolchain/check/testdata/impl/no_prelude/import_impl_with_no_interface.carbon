// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_impl_with_no_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_impl_with_no_interface.carbon

// --- lib.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: lib.carbon:[[@LINE+4]]:18: error: name `Undeclared` not found [NameNotFound]
// CHECK:STDERR: impl {.i: ()} as Undeclared {
// CHECK:STDERR:                  ^~~~~~~~~~
// CHECK:STDERR:
impl {.i: ()} as Undeclared {
  fn F() {}
}

interface Instance {
  fn G[self: Self]();
}

impl {.i: ()} as Instance {
  fn G[self: Self]() {}
}

// --- import_instance_success.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn InstanceCallImport(n: {.i: ()}) {
  n.(Instance.G)();
}

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %G.type.1e1: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.449: %G.type.1e1 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type %Instance.type [concrete]
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, @Instance.%G.decl [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.477.%G.decl) [concrete]
// CHECK:STDOUT:   %G.type.0c0: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.484: %G.type.0c0 = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.facet: %Instance.type = facet_value %struct_type.i, %impl_witness [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Undeclared = <poisoned>
// CHECK:STDOUT:     .Instance = %Instance.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.148 [concrete] {} {
// CHECK:STDOUT:     %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %Undeclared.ref: <error> = name_ref Undeclared, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Instance.decl: type = interface_decl @Instance [concrete = constants.%Instance.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.477 [concrete] {} {
// CHECK:STDOUT:     %.loc15_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %Instance.ref: type = name_ref Instance, file.%Instance.decl [concrete = constants.%Instance.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.477.%G.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %G.decl: %G.type.1e1 = fn_decl @G.1 [concrete = constants.%G.449] {
// CHECK:STDOUT:     %self.patt: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Instance.type = name_ref Self, @Instance.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %Self.ref, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.148: %struct_type.i as <error> {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.477: %struct_type.i as %Instance.ref {
// CHECK:STDOUT:   %G.decl: %G.type.0c0 = fn_decl @G.2 [concrete = constants.%G.484] {
// CHECK:STDOUT:     %self.patt: %struct_type.i = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %struct_type.i = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %struct_type.i = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.477.%struct_type.i [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     %self: %struct_type.i = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@Instance.%Self: %Instance.type) {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type.loc12_14.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %struct_type.i]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Instance.facet) {
// CHECK:STDOUT:   %Self => constants.%Instance.facet
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%struct_type.i
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %InstanceCallImport.type: type = fn_type @InstanceCallImport [concrete]
// CHECK:STDOUT:   %InstanceCallImport: %InstanceCallImport.type = struct_value () [concrete]
// CHECK:STDOUT:   %Instance.type: type = facet_type <@Instance> [concrete]
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Instance.assoc_type: type = assoc_entity_type %Instance.type [concrete]
// CHECK:STDOUT:   %assoc0: %Instance.assoc_type = assoc_entity element0, imports.%Main.import_ref.b4d [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.373) [concrete]
// CHECK:STDOUT:   %G.type.1e1: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Instance.facet: %Instance.type = facet_value %struct_type.i, %impl_witness [concrete]
// CHECK:STDOUT:   %.6d8: type = fn_type_with_self_type %G.type.1e1, %Instance.facet [concrete]
// CHECK:STDOUT:   %G.type.0c0: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.484: %G.type.0c0 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Instance: type = import_ref Main//lib, Instance, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Main.import_ref.dcd = import_ref Main//lib, inst26 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.7cb: %Instance.assoc_type = import_ref Main//lib, loc12_21, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.G = import_ref Main//lib, G, unloaded
// CHECK:STDOUT:   %Main.import_ref.536 = import_ref Main//lib, inst(ErrorInst) [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.ded207.1: type = import_ref Main//lib, loc7_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.5f0: type = import_ref Main//lib, inst(ErrorInst) [no loc], loaded [concrete = <error>]
// CHECK:STDOUT:   %Main.import_ref.97c: <witness> = import_ref Main//lib, loc15_27, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Main.import_ref.ded207.2: type = import_ref Main//lib, loc15_13, loaded [concrete = constants.%struct_type.i]
// CHECK:STDOUT:   %Main.import_ref.b49: type = import_ref Main//lib, loc15_18, loaded [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %Main.import_ref.0d8: %Instance.type = import_ref Main//lib, inst26 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Instance = imports.%Main.Instance
// CHECK:STDOUT:     .InstanceCallImport = %InstanceCallImport.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %InstanceCallImport.decl: %InstanceCallImport.type = fn_decl @InstanceCallImport [concrete = constants.%InstanceCallImport] {
// CHECK:STDOUT:     %n.patt: %struct_type.i = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %struct_type.i = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %struct_type.i = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_33: type = splice_block %struct_type.i [concrete = constants.%struct_type.i] {
// CHECK:STDOUT:       %.loc5_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc5_32.2: type = converted %.loc5_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.i: type = struct_type {.i: %empty_tuple.type} [concrete = constants.%struct_type.i]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %struct_type.i = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Instance [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.dcd
// CHECK:STDOUT:   .G = imports.%Main.import_ref.7cb
// CHECK:STDOUT:   witness = (imports.%Main.G)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.148: imports.%Main.import_ref.ded207.1 as imports.%Main.import_ref.5f0 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.536
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.477: imports.%Main.import_ref.ded207.2 as imports.%Main.import_ref.b49 [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.97c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InstanceCallImport(%n.param_patt: %struct_type.i) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %struct_type.i = name_ref n, %n
// CHECK:STDOUT:   %Instance.ref: type = name_ref Instance, imports.%Main.Instance [concrete = constants.%Instance.type]
// CHECK:STDOUT:   %G.ref: %Instance.assoc_type = name_ref G, imports.%Main.import_ref.7cb [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.6d8 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%G.484]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %bound_method(%n.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(imports.%Main.import_ref.0d8: %Instance.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %Self: %Instance.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %struct_type.i]() [from "lib.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
