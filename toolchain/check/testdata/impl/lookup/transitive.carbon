// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/transitive.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/transitive.carbon

// --- i.carbon

library "[[@TEST_NAME]]";

interface I { fn F(); }

// --- c.carbon

library "[[@TEST_NAME]]";

import library "i";

class C {}
impl C as I { fn F() {} }

// --- get.carbon

library "[[@TEST_NAME]]";

import library "c";

fn Get() -> C;

// --- call.carbon

library "[[@TEST_NAME]]";

import library "get";
import library "i";

fn Call() {
  // We find C in "get", but should look in "c" instead.
  Get().(I.F)();
}

// CHECK:STDOUT: --- i.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [template]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.5d6: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.a2e: %F.type.5d6 = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, %impl_witness [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//i, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//i, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1e0 = import_ref Main//i, loc4_21, unloaded
// CHECK:STDOUT:   %Main.F: %F.type.cf0 = import_ref Main//i, F, loaded [template = constants.%F.bc6]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//i, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "i.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%Main.import_ref.1e0
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.5d6 = fn_decl @F.2 [template = constants.%F.a2e] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5dd: %I.type) [from "i.carbon"] {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- get.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get [template]
// CHECK:STDOUT:   %Get: %Get.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//c, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//c, loc6_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//c, inst18 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Get.decl: %Get.type = fn_decl @Get [template = constants.%Get] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "c.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Get() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get [template]
// CHECK:STDOUT:   %Get: %Get.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [template]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.e03 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.742) [template]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, %impl_witness [template]
// CHECK:STDOUT:   %.076: type = fn_type_with_self_type %F.type.cf0, %I.facet [template]
// CHECK:STDOUT:   %F.type.5d6: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.a2e: %F.type.5d6 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Get: %Get.type = import_ref Main//get, Get, loaded [template = constants.%Get]
// CHECK:STDOUT:   %Main.I: type = import_ref Main//i, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//get, inst21 [indirect], loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//get, inst22 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//i, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.bcb: %I.assoc_type = import_ref Main//i, loc4_21, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %Main.F = import_ref Main//i, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.e53: <witness> = import_ref Main//c, loc7_13, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %Main.import_ref.29a: type = import_ref Main//c, loc7_6, loaded [template = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.f50: type = import_ref Main//c, loc7_11, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//i, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Get = imports.%Main.Get
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "i.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .F = imports.%Main.import_ref.bcb
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.29a as imports.%Main.import_ref.f50 [from "c.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.e53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "get.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Get.ref: %Get.type = name_ref Get, imports.%Main.Get [template = constants.%Get]
// CHECK:STDOUT:   %.loc9: ref %C = temporary_storage
// CHECK:STDOUT:   %Get.call: init %C = call %Get.ref() to %.loc9
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [template = constants.%I.type]
// CHECK:STDOUT:   %F.ref: %I.assoc_type = name_ref F, imports.%Main.import_ref.bcb [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.076 = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.a2e]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Get() -> %C [from "get.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5dd: %I.type) [from "i.carbon"] {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() [from "c.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
