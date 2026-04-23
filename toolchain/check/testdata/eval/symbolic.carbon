// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/symbolic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/symbolic.carbon

// --- symbolic_call_to_eval_fn.carbon
library "[[@TEST_NAME]]";

eval fn F(B:! bool) -> type {
  if (B) { return (); }
  return ({}, {});
}

fn G(B:! bool) {
  let unused n: F(B) = ({}, {});
}

fn H() {
  G(false);
}


// --- fail_symbolic_call_to_eval_fn_invalid_instantiation.carbon
library "[[@TEST_NAME]]";

eval fn F(B:! bool) -> type {
  if (B) { return (); }
  return ({}, {});
}

fn G(B:! bool) {
  let unused n: F(B) = ({}, {});
}

// TODO: Provide a location for the "cannot initialize" error message.
fn H() {
  // CHECK:STDERR: fail_symbolic_call_to_eval_fn_invalid_instantiation.carbon:[[@LINE+5]]:3: error: unable to monomorphize specific `G(true)` [ResolvingSpecificHere]
  // CHECK:STDERR:   G(true);
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_symbolic_call_to_eval_fn_invalid_instantiation.carbon: note: cannot initialize tuple of 0 elements from tuple with 2 elements [TupleInitElementCountMismatch]
  // CHECK:STDERR:
  G(true);
}

// --- dependent_symbolic_instruction.carbon
library "[[@TEST_NAME]]";

interface Z {}
impl {} as Z {}

eval fn E(U:! Z) -> type {
  // This makes a FacetAccessType(U) instruction, for which we have to get a
  // constant value from the definition in the called function's specific. This
  // requires that the called function has its specific definition resolved.
  //@dump-sem-ir-begin
  return U;
  //@dump-sem-ir-end
}

fn F() {
  E({});
}

// CHECK:STDOUT: --- dependent_symbolic_instruction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @empty_struct_type.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %empty_struct_type, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %U.ea3: %Z.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.9bf: type = symbolic_binding_type U, 0, %U.ea3 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b47: <witness> = impl_witness imports.%Copy.impl_witness_table.b1c [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value type, (%Copy.impl_witness.b47) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a4f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.070: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a4f, %Copy.facet [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound.16c: <bound method> = bound_method %U.binding.as_type.9bf, %type.as.Copy.impl.Op [symbolic]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound.bca: <bound method> = bound_method %empty_struct_type, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.1a7: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.b1c = impl_witness_table (%Core.import_ref.1a7), @type.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @E(%U.loc6_12.2: %Z.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc6_12.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.9bf)]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %U.binding.as_type, constants.%type.as.Copy.impl.Op [symbolic = %type.as.Copy.impl.Op.bound (constants.%type.as.Copy.impl.Op.bound.16c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: %Z.type = name_ref U, %U.loc6_12.2 [symbolic = %U.loc6_12.1 (constants.%U.ea3)]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type.9bf)]
// CHECK:STDOUT:     %.loc11: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type.9bf)]
// CHECK:STDOUT:     %impl.elem0: %.070 = impl_witness_access constants.%Copy.impl_witness.b47, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %.loc11, %impl.elem0 [symbolic = %type.as.Copy.impl.Op.bound (constants.%type.as.Copy.impl.Op.bound.16c)]
// CHECK:STDOUT:     %type.as.Copy.impl.Op.call: init type = call %bound_method(%.loc11) [symbolic = %U.binding.as_type (constants.%U.binding.as_type.9bf)]
// CHECK:STDOUT:     return %type.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E(constants.%U.ea3) {
// CHECK:STDOUT:   %U.loc6_12.1 => constants.%U.ea3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E(constants.%Z.facet) {
// CHECK:STDOUT:   %U.loc6_12.1 => constants.%Z.facet
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.binding.as_type => constants.%empty_struct_type
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound => constants.%type.as.Copy.impl.Op.bound.bca
// CHECK:STDOUT: }
// CHECK:STDOUT:
