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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_init_with_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_init_with_self.carbon

// --- fail_basic.carbon

fn Main() {
  // CHECK:STDERR: fail_basic.carbon:[[@LINE+11]]:22: error: `x` used before initialization [UsedBeforeInitialization]
  // CHECK:STDERR:   var unused x: () = x;
  // CHECK:STDERR:                      ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_basic.carbon:[[@LINE+7]]:14: error: variable `x` marked `unused` but used [UnusedButUsed]
  // CHECK:STDERR:   var unused x: () = x;
  // CHECK:STDERR:              ^~~~~
  // CHECK:STDERR: fail_basic.carbon:[[@LINE+4]]:22: note: usage here [UnusedButUsedHere]
  // CHECK:STDERR:   var unused x: () = x;
  // CHECK:STDERR:                      ^
  // CHECK:STDERR:
  var unused x: () = x;
}

// CHECK:STDOUT: --- fail_basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77: type = fn_type @DestroyImpl.WithSelf.Op.loc14 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25a: %DestroyImpl.WithSelf.Op.type.b77 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25a), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14 [concrete]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var_storage %x.var_patt
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   %.loc14_18.1: type = splice_block %.loc14_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc14_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc14_18.3: type = converted %.loc14_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc14_3.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc14_3.1: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc14_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc14_3.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc14_3.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc14_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc14(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
