// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import_struct_cycle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import_struct_cycle.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class Cycle;

var a: {.b: Cycle*};

class Cycle {
  // The type here is equivalent to the `a` above, but on import can be resolved first.
  var c: {.b: Cycle*};
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  a.b = (*a.b).c.b;
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %ptr} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.b [concrete]
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %struct_type.b [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %struct_type.b} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c [concrete]
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
// CHECK:STDOUT:     .Cycle = %Cycle.decl.loc4
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cycle.decl.loc4: type = class_decl @Cycle [concrete = constants.%Cycle] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.b = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %struct_type.b [concrete = constants.%struct_type.b] {
// CHECK:STDOUT:     %Cycle.ref: type = name_ref Cycle, %Cycle.decl.loc4 [concrete = constants.%Cycle]
// CHECK:STDOUT:     %ptr: type = ptr_type %Cycle.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %ptr} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.b = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   %Cycle.decl.loc8: type = class_decl @Cycle [concrete = constants.%Cycle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle {
// CHECK:STDOUT:   %Cycle.ref: type = name_ref Cycle, file.%Cycle.decl.loc4 [concrete = constants.%Cycle]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cycle.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %ptr} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:   %.loc10: %Cycle.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.c [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Cycle
// CHECK:STDOUT:   .Cycle = <poisoned>
// CHECK:STDOUT:   .c = %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Cycle: type = class_type @Cycle [concrete]
// CHECK:STDOUT:   %ptr.e6c: type = ptr_type %Cycle [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %ptr.e6c} [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %struct_type.b} [concrete]
// CHECK:STDOUT:   %complete_type.3fc: <witness> = complete_type_witness %struct_type.c [concrete]
// CHECK:STDOUT:   %pattern_type.d79: type = pattern_type %struct_type.b [concrete]
// CHECK:STDOUT:   %.c65: ref %ptr.e6c = struct_access imports.%a.var, element0 [concrete]
// CHECK:STDOUT:   %Cycle.elem: type = unbound_element_type %Cycle, %struct_type.b [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b7e: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%Cycle) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.6f4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Cycle) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.7c8: %ptr.as.Copy.impl.Op.type.6f4 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e6c, (%Copy.impl_witness.b7e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.706: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.e5b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.706, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.7c8, @ptr.as.Copy.impl.Op(%Cycle) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Cycle = import_ref Main//a, Cycle, unloaded
// CHECK:STDOUT:   %Main.a: ref %struct_type.b = import_ref Main//a, a, loaded [concrete = %a.var]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.741: <witness> = import_ref Main//a, loc11_1, loaded [concrete = constants.%complete_type.3fc]
// CHECK:STDOUT:   %Main.import_ref.fd1 = import_ref Main//a, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c81: %Cycle.elem = import_ref Main//a, loc10_8, loaded [concrete = %.a9f]
// CHECK:STDOUT:   %a.patt: %pattern_type.d79 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d79 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %a.var: ref %struct_type.b = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.a9f: %Cycle.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cycle = imports.%Main.Cycle
// CHECK:STDOUT:     .a = imports.%Main.a
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Cycle [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.741
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.fd1
// CHECK:STDOUT:   .c = imports.%Main.import_ref.c81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref.loc7_3: ref %struct_type.b = name_ref a, imports.%Main.a [concrete = imports.%a.var]
// CHECK:STDOUT:   %.loc7_4: ref %ptr.e6c = struct_access %a.ref.loc7_3, element0 [concrete = constants.%.c65]
// CHECK:STDOUT:   %a.ref.loc7_11: ref %struct_type.b = name_ref a, imports.%Main.a [concrete = imports.%a.var]
// CHECK:STDOUT:   %.loc7_12.1: ref %ptr.e6c = struct_access %a.ref.loc7_11, element0 [concrete = constants.%.c65]
// CHECK:STDOUT:   %.loc7_12.2: %ptr.e6c = acquire_value %.loc7_12.1
// CHECK:STDOUT:   %.loc7_10: ref %Cycle = deref %.loc7_12.2
// CHECK:STDOUT:   %c.ref: %Cycle.elem = name_ref c, imports.%Main.import_ref.c81 [concrete = imports.%.a9f]
// CHECK:STDOUT:   %.loc7_15: ref %struct_type.b = class_element_access %.loc7_10, element0
// CHECK:STDOUT:   %.loc7_17.1: ref %ptr.e6c = struct_access %.loc7_15, element0
// CHECK:STDOUT:   %.loc7_17.2: %ptr.e6c = acquire_value %.loc7_17.1
// CHECK:STDOUT:   %impl.elem0: %.e5b = impl_witness_access constants.%Copy.impl_witness.b7e, element0 [concrete = constants.%ptr.as.Copy.impl.Op.7c8]
// CHECK:STDOUT:   %bound_method.loc7_17.1: <bound method> = bound_method %.loc7_17.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Cycle) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_17.2: <bound method> = bound_method %.loc7_17.2, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.e6c = call %bound_method.loc7_17.2(%.loc7_17.2)
// CHECK:STDOUT:   assign %.loc7_4, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
