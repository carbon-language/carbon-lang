// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/initialization.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/initialization.carbon

// --- core.carbon

package Core;

interface DefaultOrUnformed {
  fn Op() -> Self;
}

interface Destroy {
  fn Op(ref self);
}

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
  var unused x: ();
  //@dump-sem-ir-end
}

// --- fail_uninitialized_no_impl.carbon

library "[[@TEST_NAME]]";

import Core;

class A {}

fn Main() {
  // CHECK:STDERR: fail_uninitialized_no_impl.carbon:[[@LINE+4]]:3: error: cannot convert type `A` into type implementing `Core.DefaultOrUnformed` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   var unused x: A;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused x: A;
}

// CHECK:STDOUT: --- initialized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt
// CHECK:STDOUT:   %.loc8_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_23.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_3: init %empty_tuple.type = converted %.loc8_23.1, %.loc8_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_18.1: type = splice_block %.loc8_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_18.3: type = converted %.loc8_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- uninitialized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %empty_tuple.type, (%DefaultOrUnformed.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.DefaultOrUnformed.impl.Op.type: type = fn_type @empty_tuple.type.as.DefaultOrUnformed.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.DefaultOrUnformed.impl.Op: %empty_tuple.type.as.DefaultOrUnformed.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9c3: %empty_tuple.type.as.DefaultOrUnformed.impl.Op.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%empty_tuple.type.as.DefaultOrUnformed.impl.Op]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @empty_tuple.type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%empty_tuple.type, (constants.%DefaultOrUnformed.impl_witness) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc8_19.1: %DefaultOrUnformed.type = converted constants.%empty_tuple.type, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc8_19.1 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc8_19.2: type = converted %.loc8_19.1, %as_type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %empty_tuple.type.as.DefaultOrUnformed.impl.Op.call: init %empty_tuple.type = call constants.%empty_tuple.type.as.DefaultOrUnformed.impl.Op()
// CHECK:STDOUT:   assign %x.var, %empty_tuple.type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc8_18.1: type = splice_block %.loc8_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_18.3: type = converted %.loc8_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
