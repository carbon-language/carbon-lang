// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/initialization.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/initialization.carbon

// --- core.carbon

package Core;

// CHECK:STDERR: core.carbon:[[@LINE+7]]:1: error: redeclaration of `interface DefaultOrUnformed` is redundant [RedeclRedundant]
// CHECK:STDERR: interface DefaultOrUnformed {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: min_prelude/parts/default.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: interface DefaultOrUnformed { fn Op() -> Self; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface DefaultOrUnformed {
  fn Op() -> Self;
}

// CHECK:STDERR: core.carbon:[[@LINE+10]]:1: error: orphan `impl` found; something in the self-type or constraint must be defined in the same file [ImplIsOrphan]
// CHECK:STDERR: impl () as DefaultOrUnformed {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: min_prelude/parts/default.carbon:4:1: error: duplicate name `DefaultOrUnformed` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: interface DefaultOrUnformed { fn Op() -> Self; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: core.carbon:[[@LINE-11]]:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: interface DefaultOrUnformed {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl () as DefaultOrUnformed {
  fn Op() -> Self { return (); }
}

// --- initialized.carbon

library "[[@TEST_NAME]]";

import Core;

fn Main() {
  //@dump-sem-ir-begin
  var unused x: () = ();
  //@dump-sem-ir-end
}

// --- uninitialized.carbon

library "[[@TEST_NAME]]";

import Core;

fn Main() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: uninitialized.carbon:[[@LINE+14]]:3: note: in name lookup for `DefaultOrUnformed` [InNameLookup]
  // CHECK:STDERR:   var unused x: ();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: uninitialized.carbon:[[@LINE+10]]:3: error: cannot convert type `()` into type implementing `Core.DefaultOrUnformed` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   var unused x: ();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: min_prelude/parts/default.carbon:4:1: error: duplicate name `DefaultOrUnformed` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: interface DefaultOrUnformed { fn Op() -> Self; }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:11:1: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR: interface DefaultOrUnformed {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  var unused x: ();
  //@dump-sem-ir-end
}

// --- fail_uninitialized_no_impl.carbon

library "[[@TEST_NAME]]";

import Core;

class A {}

fn Main() {
  // CHECK:STDERR: fail_uninitialized_no_impl.carbon:[[@LINE+8]]:3: note: in name lookup for `DefaultOrUnformed` [InNameLookup]
  // CHECK:STDERR:   var unused x: A;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_uninitialized_no_impl.carbon:[[@LINE+4]]:3: error: cannot convert type `A` into type implementing `Core.DefaultOrUnformed` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   var unused x: A;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused x: A;
}

// CHECK:STDOUT: --- initialized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77: type = fn_type @DestroyImpl.WithSelf.Op.loc21 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25a: %DestroyImpl.WithSelf.Op.type.b77 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25a), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc21 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %empty_tuple.type, (%custom_witness.2cb, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var_storage %x.var_patt
// CHECK:STDOUT:   %.loc21_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21_23.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21_3.1: init %empty_tuple.type = converted %.loc21_23.1, %.loc21_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var, %.loc21_3.1
// CHECK:STDOUT:   %.loc21_18.1: type = splice_block %.loc21_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc21_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc21_18.3: type = converted %.loc21_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc21_3.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc21_3.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc21_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc21_3.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc21_3.3: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc21_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc21(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc21(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- uninitialized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77: type = fn_type @DestroyImpl.WithSelf.Op.loc29 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25a: %DestroyImpl.WithSelf.Op.type.b77 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25a), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc29 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %empty_tuple.type, (%custom_witness.2cb, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var_storage %x.var_patt
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   %.loc29_18.1: type = splice_block %.loc29_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc29_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc29_18.3: type = converted %.loc29_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc29_3.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc29_3.1: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc29_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc29_3.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc29_3.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc29_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc29(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc29(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
