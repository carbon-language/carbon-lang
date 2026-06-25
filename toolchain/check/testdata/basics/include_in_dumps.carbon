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
  fn Op(self);
}
//@dump-sem-ir-end

class C {
  impl C as I {
    fn Op(unused self) {}
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
  fn Op(self);
}
//@dump-sem-ir-end

class C {
  impl C as I {
    fn Op(unused self) {}
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- exclude/included_with_range.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.c60: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.a33: %pattern_type.c60 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.696: %pattern_type.c60 = at_binding_pattern self, %self.param_patt.a33 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.type.dce: type = fn_type @I.WithSelf.Op, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.c70: %I.WithSelf.Op.type.dce = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.49a: %pattern_type.98b = at_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Op.type.194: type = fn_type @I.WithSelf.Op, @I.WithSelf(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Op.95d: %I.WithSelf.Op.type.194 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.Op.decl: @I.WithSelf.%I.WithSelf.Op.type (%I.WithSelf.Op.type.dce) = fn_decl @I.WithSelf.Op [symbolic = @I.WithSelf.%I.WithSelf.Op (constants.%I.WithSelf.Op.c70)] {
// CHECK:STDOUT:     %self.param_patt.loc8_9.1: @I.WithSelf.Op.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc8_9.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:     %self.patt.loc8_9.1: @I.WithSelf.Op.%pattern_type (%pattern_type.c60) = at_binding_pattern self, %self.param_patt.loc8_9.1 [symbolic = %self.patt.loc8_9.2 (constants.%self.patt.696)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.Op.%Self.as_type.loc8_9.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_9.1: type = splice_block %.loc8_9.2 [symbolic = %Self.as_type.loc8_9.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc8_9.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc8_9.2: type = converted %Self.ref, %Self.as_type.loc8_9.2 [symbolic = %Self.as_type.loc8_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.Op.%Self.as_type.loc8_9.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.Op(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc8_9.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_9.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc8_9.1 [symbolic = %pattern_type (constants.%pattern_type.c60)]
// CHECK:STDOUT:   %self.param_patt.loc8_9.2: @I.WithSelf.Op.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc8_9.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:   %self.patt.loc8_9.2: @I.WithSelf.Op.%pattern_type (%pattern_type.c60) = at_binding_pattern self, %self.param_patt.loc8_9.2 [symbolic = %self.patt.loc8_9.2 (constants.%self.patt.696)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.Op.%Self.as_type.loc8_9.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.Op.type => constants.%I.WithSelf.Op.type.dce
// CHECK:STDOUT:   %I.WithSelf.Op => constants.%I.WithSelf.Op.c70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc8_9.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c60
// CHECK:STDOUT:   %self.param_patt.loc8_9.2 => constants.%self.param_patt.a33
// CHECK:STDOUT:   %self.patt.loc8_9.2 => constants.%self.patt.696
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.Op.type => constants.%I.WithSelf.Op.type.194
// CHECK:STDOUT:   %I.WithSelf.Op => constants.%I.WithSelf.Op.95d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Op(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc8_9.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %self.param_patt.loc8_9.2 => constants.%self.param_patt.303
// CHECK:STDOUT:   %self.patt.loc8_9.2 => constants.%self.patt.49a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_included.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.type.351: type = fn_type @I.WithSelf.Op, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.fcf: %I.WithSelf.Op.type.351 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %self.param_patt.58e: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.300: %pattern_type.b3a = at_binding_pattern self, %self.param_patt.58e [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.421 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Op.type.06d: type = fn_type @I.WithSelf.Op, @I.WithSelf(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Op.c74: %I.WithSelf.Op.type.06d = struct_value () [concrete]
// CHECK:STDOUT:   %.536: type = fn_type_with_self_type %I.WithSelf.Op.type.06d, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.Op.type: type = fn_type @C.as.I.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.I.impl.Op: %C.as.I.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//included_with_range, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//included_with_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//included_with_range, loc16_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//included_with_range, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.ddc: %I.assoc_type = import_ref Main//included_with_range, loc8_14, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.Op = import_ref Main//included_with_range, Op, unloaded
// CHECK:STDOUT:   %Main.import_ref.e59c42.2: %I.type = import_ref Main//included_with_range, loc7_13, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//included_with_range, loc7_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.421: @I.WithSelf.%I.WithSelf.Op.type (%I.WithSelf.Op.type.351) = import_ref Main//included_with_range, loc8_14, loaded [symbolic = @I.WithSelf.%I.WithSelf.Op (constants.%I.WithSelf.Op.fcf)]
// CHECK:STDOUT:   %Main.import_ref.b48: <witness> = import_ref Main//included_with_range, loc13_15, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.335: type = import_ref Main//included_with_range, loc13_8, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.c3f: type = import_ref Main//included_with_range, loc13_13, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.I.impl.Op.type = import_ref Main//included_with_range, loc14_24, loaded [concrete = constants.%C.as.I.impl.Op]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.341), @C.as.I.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .Op = imports.%Main.import_ref.ddc
// CHECK:STDOUT:   witness = (imports.%Main.Op)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%Main.import_ref.335 as imports.%Main.import_ref.c3f [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.b48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %Op.ref: %I.assoc_type = name_ref Op, imports.%Main.import_ref.ddc [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.536 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.I.impl.Op.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.Op(imports.%Main.import_ref.e59c42.2: %I.type) [from "exclude/included_with_range.carbon"] {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.b3a)]
// CHECK:STDOUT:   %self.param_patt: @I.WithSelf.Op.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.58e)]
// CHECK:STDOUT:   %self.patt: @I.WithSelf.Op.%pattern_type (%pattern_type.b3a) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.300)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.Op [from "exclude/included_with_range.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %I.WithSelf.Op.type => constants.%I.WithSelf.Op.type.351
// CHECK:STDOUT:   %I.WithSelf.Op => constants.%I.WithSelf.Op.fcf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Op(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.58e
// CHECK:STDOUT:   %self.patt => constants.%self.patt.300
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.Op.type => constants.%I.WithSelf.Op.type.06d
// CHECK:STDOUT:   %I.WithSelf.Op => constants.%I.WithSelf.Op.c74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_excluded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.type.351: type = fn_type @I.WithSelf.Op, @I.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Op.fcf: %I.WithSelf.Op.type.351 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.421 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Op.type.06d: type = fn_type @I.WithSelf.Op, @I.WithSelf(%I.facet) [concrete]
// CHECK:STDOUT:   %.536: type = fn_type_with_self_type %I.WithSelf.Op.type.06d, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.Op.type: type = fn_type @C.as.I.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.I.impl.Op: %C.as.I.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//excluded_with_range, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//excluded_with_range, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.ddc: %I.assoc_type = import_ref Main//excluded_with_range, loc6_14, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.import_ref.421: @I.WithSelf.%I.WithSelf.Op.type (%I.WithSelf.Op.type.351) = import_ref Main//excluded_with_range, loc6_14, loaded [symbolic = @I.WithSelf.%I.WithSelf.Op (constants.%I.WithSelf.Op.fcf)]
// CHECK:STDOUT:   %Main.import_ref.341: %C.as.I.impl.Op.type = import_ref Main//excluded_with_range, loc12_24, loaded [concrete = constants.%C.as.I.impl.Op]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.341), @C.as.I.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %Op.ref: %I.assoc_type = name_ref Op, imports.%Main.import_ref.ddc [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.536 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.I.impl.Op.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
