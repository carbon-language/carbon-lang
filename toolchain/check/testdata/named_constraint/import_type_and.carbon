// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/import_type_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/import_type_and.carbon

// --- import.carbon
library "[[@TEST_NAME]]";
//@include-in-dumps

interface Y {}
interface Z {}

constraint N {
  // Tests importing and formatting the Call instruction for the `&` operation.
  require impls Y & Z;
}

// --- import.impl.carbon
impl library "[[@TEST_NAME]]";

//@include-in-dumps

fn F(_:! N) {}

// CHECK:STDOUT: --- import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.765: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.cb6: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %Self.d87: %N.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.138: type = facet_access_type %Self.d87 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.38f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d56: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.38f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Y.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Y & @Z> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %N.decl: type = constraint_decl @N [concrete = constants.%N.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.765]
// CHECK:STDOUT:   %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.cb6]
// CHECK:STDOUT:   %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @N {
// CHECK:STDOUT:   %Self: %N.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d87]
// CHECK:STDOUT:   %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %N.WithSelf.Self.as_type.impls.facet_type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.facet_type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc9_11.1 impls %type.as.BitAndWith.impl.Op.call
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc9_11.1: type = facet_access_type @N.%Self [symbolic = %Self.as_type.loc9_11.2 (constants.%Self.as_type.138)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %impl.elem0: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %Y.ref, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:     %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%Y.ref, %Z.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:   .Y = <poisoned>
// CHECK:STDOUT:   .Z = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @N.WithSelf.Self.as_type.impls.facet_type.require {
// CHECK:STDOUT:     require @N.WithSelf.Self.as_type.impls.facet_type.require.%Self.as_type.loc9_11.1 impls @N.WithSelf.Self.as_type.impls.facet_type.require.%type.as.BitAndWith.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.facet_type.require(@N.%Self: %N.type) {
// CHECK:STDOUT:   %Self: %N.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d87)]
// CHECK:STDOUT:   %Self.as_type.loc9_11.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_11.2 (constants.%Self.as_type.138)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.765) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.cb6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%Self.d87) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.facet_type.require(constants.%Self.d87) {
// CHECK:STDOUT:   %Self => constants.%Self.d87
// CHECK:STDOUT:   %Self.as_type.loc9_11.2 => constants.%Self.as_type.138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %Self.f45: %N.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.013: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.95b: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Z & @Y> [concrete]
// CHECK:STDOUT:   %Self.as_type.190: type = facet_access_type %Self.f45 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type %N.type [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.9a5 = symbolic_binding_pattern _, 0 [symbolic]
// CHECK:STDOUT:   %_: %N.type = symbolic_binding _, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Y = import_ref Main//import, Y, unloaded
// CHECK:STDOUT:   %Main.Z = import_ref Main//import, Z, unloaded
// CHECK:STDOUT:   %Main.N: type = import_ref Main//import, N, loaded [concrete = constants.%N.type]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.e73 = import_ref Main//import, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.28a = import_ref Main//import, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.5f6: type = import_ref Main//import, loc9_11, loaded [symbolic = @N.WithSelf.Self.as_type.impls.facet_type.require.%Self.as_type (constants.%Self.as_type.190)]
// CHECK:STDOUT:   %Main.import_ref.624: init type = import_ref Main//import, loc9_19, loaded [concrete = constants.%facet_type]
// CHECK:STDOUT:   %Main.import_ref.423e3a.2: %N.type = import_ref Main//import, loc7_14, loaded [symbolic = constants.%Self.f45]
// CHECK:STDOUT:   %Main.import_ref.364 = import_ref Main//import, loc7_14, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Y = imports.%Main.Y
// CHECK:STDOUT:     .Z = imports.%Main.Z
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_22.1 = import <none>
// CHECK:STDOUT:   %default.import.loc1_22.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %_.patt.loc5_7.1: %pattern_type.9a5 = symbolic_binding_pattern _, 0 [symbolic = %_.patt.loc5_7.2 (constants.%_.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %N.ref [concrete = constants.%N.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %N.ref: type = name_ref N, imports.%Main.N [concrete = constants.%N.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_.loc5_7.2: %N.type = symbolic_binding _, 0 [symbolic = %_.loc5_7.1 (constants.%_)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z [from "import.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e73
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y [from "import.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.28a
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: constraint @N [from "import.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.364
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @N.WithSelf.Self.as_type.impls.facet_type.require {
// CHECK:STDOUT:     require imports.%Main.import_ref.5f6 impls imports.%Main.import_ref.624
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.facet_type.require(imports.%Main.import_ref.423e3a.2: %N.type) [from "import.carbon"] {
// CHECK:STDOUT:   %Self: %N.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f45)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.190)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%_.loc5_7.2: %N.type) {
// CHECK:STDOUT:   %_.patt.loc5_7.2: %pattern_type.9a5 = symbolic_binding_pattern _, 0 [symbolic = %_.patt.loc5_7.2 (constants.%_.patt)]
// CHECK:STDOUT:   %_.loc5_7.1: %N.type = symbolic_binding _, 0 [symbolic = %_.loc5_7.1 (constants.%_)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%Self.f45) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self.013) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.95b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.facet_type.require(constants.%Self.f45) {
// CHECK:STDOUT:   %Self => constants.%Self.f45
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%_) {
// CHECK:STDOUT:   %_.patt.loc5_7.2 => constants.%_.patt
// CHECK:STDOUT:   %_.loc5_7.1 => constants.%_
// CHECK:STDOUT: }
// CHECK:STDOUT:
