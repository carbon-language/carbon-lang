// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/return_implicit.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/return_implicit.carbon

fn MakeImplicitEmptyTuple() {
}

fn Main() {
  var unused b: () = MakeImplicitEmptyTuple();
}

// CHECK:STDOUT: --- return_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %MakeImplicitEmptyTuple.type: type = fn_type @MakeImplicitEmptyTuple [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MakeImplicitEmptyTuple: %MakeImplicitEmptyTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77: type = fn_type @DestroyImpl.WithSelf.Op.loc19 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25a: %DestroyImpl.WithSelf.Op.type.b77 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25a), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc19 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %empty_tuple.type, (%custom_witness.2cb, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .MakeImplicitEmptyTuple = %MakeImplicitEmptyTuple.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %MakeImplicitEmptyTuple.decl: %MakeImplicitEmptyTuple.type = fn_decl @MakeImplicitEmptyTuple [concrete = constants.%MakeImplicitEmptyTuple] {} {}
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeImplicitEmptyTuple() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var_storage %b.var_patt
// CHECK:STDOUT:   %MakeImplicitEmptyTuple.ref: %MakeImplicitEmptyTuple.type = name_ref MakeImplicitEmptyTuple, file.%MakeImplicitEmptyTuple.decl [concrete = constants.%MakeImplicitEmptyTuple]
// CHECK:STDOUT:   %MakeImplicitEmptyTuple.call: init %empty_tuple.type = call %MakeImplicitEmptyTuple.ref()
// CHECK:STDOUT:   assign %b.var, %MakeImplicitEmptyTuple.call
// CHECK:STDOUT:   %.loc19_18.1: type = splice_block %.loc19_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc19_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc19_18.3: type = converted %.loc19_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc19_3.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc19_3.1: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc19_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc19_3.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc19_3.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc19_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%b.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc19(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc19(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
