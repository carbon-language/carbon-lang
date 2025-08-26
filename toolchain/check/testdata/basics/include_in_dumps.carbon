// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// This explicitly excludes some files, then set ranges to `if-present` so that
// we can see imports (particularly of `impl`s) from files that would be excluded without
// `//@include-in-dumps`.
// EXTRA-ARGS: --exclude-dump-file-prefix=exclude/ --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/include_in_dumps.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/include_in_dumps.carbon

// --- exclude/included.carbon

library "[[@TEST_NAME]]";

//@include-in-dumps

interface I {}

// --- exclude/included_with_range.carbon

library "[[@TEST_NAME]]";

//@include-in-dumps

//@dump-sem-ir-begin
interface I {
  fn Op[self: Self]();
}
//@dump-sem-ir-end

class C {
  impl C as I {
    fn Op[self: Self]() {}
  }
}

// --- import_included.carbon

library "[[@TEST_NAME]]";

import library "included_with_range";

fn F(c: C) { c.(I.Op)(); }

// --- exclude/excluded.carbon

library "[[@TEST_NAME]]";

interface I {}

// --- exclude/excluded_with_range.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface I {
  fn Op[self: Self]();
}
//@dump-sem-ir-end

class C {
  impl C as I {
    fn Op[self: Self]() {}
  }
}

// --- import_excluded.carbon

library "[[@TEST_NAME]]";

import library "excluded_with_range";

fn F(c: C) { c.(I.Op)(); }

// CHECK:STDOUT: --- exclude/included.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
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
// CHECK:STDOUT: --- exclude/included_with_range.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %I.Op.type: type = fn_type @I.Op [concrete]
// CHECK:STDOUT:   %I.Op: %I.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.Op.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.Op.decl: %I.Op.type = fn_decl @I.Op [concrete = constants.%I.Op] {
// CHECK:STDOUT:     %self.patt: @I.Op.%pattern_type (%pattern_type.6de) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.Op.%pattern_type (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.Op.%Self.as_type.loc8_15.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc8_15.1: type = splice_block %.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc8_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc8_15.2: type = converted %Self.ref, %Self.as_type.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.Op.%Self.as_type.loc8_15.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%I.Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.Op(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc8_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc8_15.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.Op.%Self.as_type.loc8_15.1 (%Self.as_type.b70));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.Op(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.Op(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_included.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.7d9 [concrete]
// CHECK:STDOUT:   %I.Op.type: type = fn_type @I.Op [concrete]
// CHECK:STDOUT:   %I.Op: %I.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %.36a: type = fn_type_with_self_type %I.Op.type, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.Op.type: type = fn_type @C.as.I.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.I.impl.Op: %C.as.I.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//included_with_range, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//included_with_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//included_with_range, loc16_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//included_with_range, inst43 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//included_with_range, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.9cd: %I.assoc_type = import_ref Main//included_with_range, loc8_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.Op.ca5 = import_ref Main//included_with_range, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.7d9: %I.Op.type = import_ref Main//included_with_range, loc8_22, loaded [concrete = constants.%I.Op]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//included_with_range, inst19 [no loc], loaded [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %Main.import_ref.d7a: <witness> = import_ref Main//included_with_range, loc13_15, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.29a: type = import_ref Main//included_with_range, loc13_8, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.301: type = import_ref Main//included_with_range, loc13_13, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.34c = import_ref Main//included_with_range, loc12_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.0ed: type = import_ref Main//included_with_range, loc12_9, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.cb9: type = import_ref Main//included_with_range, inst62 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.003: %C.as.I.impl.Op.type = import_ref Main//included_with_range, loc14_25, loaded [concrete = constants.%C.as.I.impl.Op]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.003), @C.as.I.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c48 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.9cd
// CHECK:STDOUT:   witness = (imports.%Main.Op.ca5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%Main.import_ref.29a as imports.%Main.import_ref.301 [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.d7a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: imports.%Main.import_ref.0ed as imports.%Main.import_ref.cb9 [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.34c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %Op.ref: %I.assoc_type = name_ref Op, imports.%Main.import_ref.9cd [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.36a = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.I.impl.Op.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.Op(imports.%Main.import_ref.5dd: %I.type) [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.Op [from "exclude/included_with_range.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.Op(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_excluded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.7d9 [concrete]
// CHECK:STDOUT:   %I.Op.type: type = fn_type @I.Op [concrete]
// CHECK:STDOUT:   %I.Op: %I.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %.36a: type = fn_type_with_self_type %I.Op.type, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.Op.type: type = fn_type @C.as.I.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.I.impl.Op: %C.as.I.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//excluded_with_range, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//excluded_with_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.9cd: %I.assoc_type = import_ref Main//excluded_with_range, loc6_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.import_ref.7d9: %I.Op.type = import_ref Main//excluded_with_range, loc6_22, loaded [concrete = constants.%I.Op]
// CHECK:STDOUT:   %Main.import_ref.003: %C.as.I.impl.Op.type = import_ref Main//excluded_with_range, loc12_25, loaded [concrete = constants.%C.as.I.impl.Op]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.003), @C.as.I.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c48 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %Op.ref: %I.assoc_type = name_ref Op, imports.%Main.import_ref.9cd [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.36a = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.I.impl.Op.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
