// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_extend_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_extend_impl.carbon

// --- extend_impl_library.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F();
}

class C {
  extend impl as I {
    fn F() {}
  }
}

// --- use_imported_class_extend_impl.carbon

library "[[@TEST_NAME]]";

import library "extend_impl_library";

fn G(c: C) {
  C.F();
  c.F();
}

// CHECK:STDOUT: --- extend_impl_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type.cf0 [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.f36: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.4c3: %F.type.f36 = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, %impl_witness [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [template = constants.%F.bc6] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.f36 = fn_decl @F.2 [template = constants.%F.4c3] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:
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
// CHECK:STDOUT: --- use_imported_class_extend_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type.cf0 [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, imports.%import_ref.777 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.f6d) [template]
// CHECK:STDOUT:   %F.type.f36: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.4c3: %F.type.f36 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.541 = import_ref Main//extend_impl_library, I, unloaded
// CHECK:STDOUT:   %import_ref.3b0: type = import_ref Main//extend_impl_library, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref Main//extend_impl_library, loc12_1, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//extend_impl_library, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.b5b: %F.assoc_type = import_ref Main//extend_impl_library, loc5_9, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.8b9 = import_ref Main//extend_impl_library, F, unloaded
// CHECK:STDOUT:   %import_ref.b86: <witness> = import_ref Main//extend_impl_library, loc9_20, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.0ed: type = import_ref Main//extend_impl_library, loc9_15, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3019d1.2: type = import_ref Main//extend_impl_library, loc9_18, loaded [template = constants.%I.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.541
// CHECK:STDOUT:     .C = imports.%import_ref.3b0
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "extend_impl_library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .F = imports.%import_ref.b5b
// CHECK:STDOUT:   witness = (imports.%import_ref.8b9)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.0ed as imports.%import_ref.3019d1.2 [from "extend_impl_library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.b86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "extend_impl_library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc7: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   %F.ref.loc7: %F.assoc_type = name_ref F, imports.%import_ref.b5b [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc7: %F.type.cf0 = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.4c3]
// CHECK:STDOUT:   %F.call.loc7: init %empty_tuple.type = call %impl.elem0.loc7()
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc8: %F.assoc_type = name_ref F, imports.%import_ref.b5b [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc8: %F.type.cf0 = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.4c3]
// CHECK:STDOUT:   %F.call.loc8: init %empty_tuple.type = call %impl.elem0.loc8()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self: %I.type) [from "extend_impl_library.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() [from "extend_impl_library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
