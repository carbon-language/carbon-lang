// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/min_prelude/tuple_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/min_prelude/tuple_pattern.carbon


// --- basic.carbon

library "[[@TEST_NAME]]";

fn F() {
  let (x: {}, y: {}) = ({}, {});
  var (a: {}, b: {});
  var (c: {}, d: {}) = ({}, {});
}

// --- variable.carbon

library "[[@TEST_NAME]]";

fn F() {
  var tuple: ({}, {}) = ({}, {});
  var (x: {}, y: {}) = tuple;
}

fn G() {
  let tuple: ({}, {}) = ({}, {});
  var (x: {}, y: {}) = tuple;
}

fn MakeTuple() -> ({}, {});

fn H() {
  var (x: {}, y: {}) = MakeTuple();
}

// --- nested.carbon

library "[[@TEST_NAME]]";

fn F() {
  let (x: {}, (y: {}, z: {})) = ({}, ({}, {}));
}

// --- package_scope.carbon

library "[[@TEST_NAME]]";

let (x: {}, y: {}) = ({}, {});

// --- fail_in_interface.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_in_interface.carbon:[[@LINE+4]]:7: error: pattern in associated constant declaration must be a single `:!` binding [ExpectedSymbolicBindingInAssociatedConstant]
  // CHECK:STDERR:   let (x: {}, y: {});
  // CHECK:STDERR:       ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let (x: {}, y: {});
}

// --- fail_in_class.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_in_class.carbon:[[@LINE+8]]:7: error: expected identifier in field declaration [ExpectedFieldIdentifier]
  // CHECK:STDERR:   var (x: {}, y: {});
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_in_class.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var (x: {}, y: {});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var (x: {}, y: {});
}

// --- fail_initializer_mismatch.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_initializer_mismatch.carbon:[[@LINE+4]]:5: error: tuple pattern expects 2 elements, but tuple literal has 1 [TuplePatternSizeDoesntMatchLiteral]
// CHECK:STDERR: let (x: {}, y: {}) = ({},);
// CHECK:STDERR:     ^~~~~~~~~~~~~~
// CHECK:STDERR:
let (x: {}, y: {}) = ({},);

// CHECK:STDERR: fail_initializer_mismatch.carbon:[[@LINE+4]]:5: error: tuple pattern expects 2 elements, but tuple literal has 3 [TuplePatternSizeDoesntMatchLiteral]
// CHECK:STDERR: let (a: {}, b: {}) = ({}, {}, {});
// CHECK:STDERR:     ^~~~~~~~~~~~~~
// CHECK:STDERR:
let (a: {}, b: {}) = ({}, {}, {});

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.725 [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.706: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.9c2: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %Op.type.72a: type = fn_type @Op.2, @impl(%tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %Op.973: %Op.type.72a = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %tuple.type.b6b [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type.b6b, (%Destroy.impl_witness.9c2) [concrete]
// CHECK:STDOUT:   %.8ce: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.973, @Op.2(%tuple.type.b6b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.40f = import_ref Core//prelude, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.f99: %Destroy.assoc_type = import_ref Core//prelude, loc9_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.150: %Destroy.type = import_ref Core//prelude, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Core.import_ref.4b4: <witness> = import_ref Core//prelude, loc11_37, loaded [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc11_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.56e: type = import_ref Core//prelude, loc11_29, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Destroy [from "include_files/destroy.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.40f
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness constants.%Destroy.impl_witness_table, @impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.bc9)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.bc9) = struct_value () [symbolic = %Op (constants.%Op.46f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.56e {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.4b4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc5_20: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_31: %tuple.type.b6b = tuple_literal (%.loc5_26.1, %.loc5_30.1)
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_26: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_26.2: %empty_struct_type = converted %.loc5_26.1, %empty_struct.loc5_26 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = bind_name x, %.loc5_26.2
// CHECK:STDOUT:   %.loc5_19.1: type = splice_block %.loc5_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_19.3: type = converted %.loc5_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_30: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_30.2: %empty_struct_type = converted %.loc5_30.1, %empty_struct.loc5_30 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = bind_name y, %.loc5_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_struct_type = binding_pattern a
// CHECK:STDOUT:     %b.patt: %empty_struct_type = binding_pattern b
// CHECK:STDOUT:     %.loc6_20: %tuple.type.b6b = tuple_pattern (%a.patt, %b.patt)
// CHECK:STDOUT:     %.loc6_3.1: %tuple.type.b6b = var_pattern %.loc6_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %impl.elem0.loc6: %.8ce = impl_witness_access constants.%Destroy.impl_witness.9c2, element0 [concrete = constants.%Op.973]
// CHECK:STDOUT:   %bound_method.loc6: <bound method> = bound_method %.var.loc6, %impl.elem0.loc6
// CHECK:STDOUT:   %tuple.elem0.loc6_3.1: ref %empty_struct_type = tuple_access %.var.loc6, element0
// CHECK:STDOUT:   %tuple.elem1.loc6_3.1: ref %empty_struct_type = tuple_access %.var.loc6, element1
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_struct_type = bind_name a, %tuple.elem0.loc6_3.1
// CHECK:STDOUT:   %.loc6_19.1: type = splice_block %.loc6_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_19.3: type = converted %.loc6_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_struct_type = bind_name b, %tuple.elem1.loc6_3.1
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %empty_struct_type = binding_pattern c
// CHECK:STDOUT:     %d.patt: %empty_struct_type = binding_pattern d
// CHECK:STDOUT:     %.loc7_20: %tuple.type.b6b = tuple_pattern (%c.patt, %d.patt)
// CHECK:STDOUT:     %.loc7_3.1: %tuple.type.b6b = var_pattern %.loc7_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc7: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %impl.elem0.loc7: %.8ce = impl_witness_access constants.%Destroy.impl_witness.9c2, element0 [concrete = constants.%Op.973]
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %.var.loc7, %impl.elem0.loc7
// CHECK:STDOUT:   %.loc7_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_31.1: %tuple.type.b6b = tuple_literal (%.loc7_26.1, %.loc7_30.1)
// CHECK:STDOUT:   %tuple.elem0.loc7_31: ref %empty_struct_type = tuple_access %.var.loc7, element0
// CHECK:STDOUT:   %.loc7_26.2: init %empty_struct_type = struct_init () to %tuple.elem0.loc7_31 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_31.2: init %empty_struct_type = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc7_31: ref %empty_struct_type = tuple_access %.var.loc7, element1
// CHECK:STDOUT:   %.loc7_30.2: init %empty_struct_type = struct_init () to %tuple.elem1.loc7_31 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_31.3: init %empty_struct_type = converted %.loc7_30.1, %.loc7_30.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_31.4: init %tuple.type.b6b = tuple_init (%.loc7_31.2, %.loc7_31.3) to %.var.loc7 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc7_3.2: init %tuple.type.b6b = converted %.loc7_31.1, %.loc7_31.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var.loc7, %.loc7_3.2
// CHECK:STDOUT:   %tuple.elem0.loc7_3.1: ref %empty_struct_type = tuple_access %.var.loc7, element0
// CHECK:STDOUT:   %tuple.elem1.loc7_3.1: ref %empty_struct_type = tuple_access %.var.loc7, element1
// CHECK:STDOUT:   %.loc7_12.1: type = splice_block %.loc7_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc7_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_12.3: type = converted %.loc7_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_struct_type = bind_name c, %tuple.elem0.loc7_3.1
// CHECK:STDOUT:   %.loc7_19.1: type = splice_block %.loc7_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc7_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_19.3: type = converted %.loc7_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_struct_type = bind_name d, %tuple.elem1.loc7_3.1
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Op.2(constants.%tuple.type.b6b) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.var.loc7, %specific_fn.loc7
// CHECK:STDOUT:   %tuple.elem0.loc7_3.2: ref %empty_struct_type = tuple_access %.var.loc7, element0
// CHECK:STDOUT:   %empty_struct.loc7_3.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_3.3: %empty_struct_type = converted %tuple.elem0.loc7_3.2, %empty_struct.loc7_3.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc7_3.2: ref %empty_struct_type = tuple_access %.var.loc7, element1
// CHECK:STDOUT:   %empty_struct.loc7_3.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_3.4: %empty_struct_type = converted %tuple.elem1.loc7_3.2, %empty_struct.loc7_3.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc7: %tuple.type.b6b = tuple_value (%.loc7_3.3, %.loc7_3.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc7_3.5: %tuple.type.b6b = converted %.var.loc7, %tuple.loc7 [concrete = constants.%tuple]
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc7_3.5)
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Op.2(constants.%tuple.type.b6b) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.var.loc6, %specific_fn.loc6
// CHECK:STDOUT:   %tuple.elem0.loc6_3.2: ref %empty_struct_type = tuple_access %.var.loc6, element0
// CHECK:STDOUT:   %empty_struct.loc6_3.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_3.2: %empty_struct_type = converted %tuple.elem0.loc6_3.2, %empty_struct.loc6_3.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc6_3.2: ref %empty_struct_type = tuple_access %.var.loc6, element1
// CHECK:STDOUT:   %empty_struct.loc6_3.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_3.3: %empty_struct_type = converted %tuple.elem1.loc6_3.2, %empty_struct.loc6_3.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc6: %tuple.type.b6b = tuple_value (%.loc6_3.2, %.loc6_3.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_3.4: %tuple.type.b6b = converted %.var.loc6, %tuple.loc6 [concrete = constants.%tuple]
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc6_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.150: %Destroy.type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.706
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%tuple.type.b6b) {
// CHECK:STDOUT:   %T => constants.%tuple.type.b6b
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.9c2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.72a
// CHECK:STDOUT:   %Op => constants.%Op.973
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%tuple.type.b6b) {
// CHECK:STDOUT:   %T => constants.%tuple.type.b6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- variable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.725 [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.706: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.9c2: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %Op.type.72a: type = fn_type @Op.2, @impl(%tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %Op.973: %Op.type.72a = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %tuple.type.b6b [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type.b6b, (%Destroy.impl_witness.9c2) [concrete]
// CHECK:STDOUT:   %.8ce: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.973, @Op.2(%tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeTuple.type: type = fn_type @MakeTuple [concrete]
// CHECK:STDOUT:   %MakeTuple: %MakeTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.40f = import_ref Core//prelude, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.f99: %Destroy.assoc_type = import_ref Core//prelude, loc9_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.150: %Destroy.type = import_ref Core//prelude, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Core.import_ref.4b4: <witness> = import_ref Core//prelude, loc11_37, loaded [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc11_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.56e: type = import_ref Core//prelude, loc11_29, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .MakeTuple = %MakeTuple.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %MakeTuple.decl: %MakeTuple.type = fn_decl @MakeTuple [concrete = constants.%MakeTuple] {
// CHECK:STDOUT:     %return.patt: %tuple.type.b6b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.b6b = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_21: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_25: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_26.1: %tuple.type.b6b = tuple_literal (%.loc14_21, %.loc14_25)
// CHECK:STDOUT:     %.loc14_26.2: type = converted %.loc14_21, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc14_26.3: type = converted %.loc14_25, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc14_26.4: type = converted %.loc14_26.1, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:     %return.param: ref %tuple.type.b6b = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.b6b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Destroy [from "include_files/destroy.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.40f
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness constants.%Destroy.impl_witness_table, @impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.bc9)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.bc9) = struct_value () [symbolic = %Op (constants.%Op.46f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.56e {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.4b4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple.patt: %tuple.type.b6b = binding_pattern tuple
// CHECK:STDOUT:     %.loc5_3.1: %tuple.type.b6b = var_pattern %tuple.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.var: ref %tuple.type.b6b = var tuple
// CHECK:STDOUT:   %impl.elem0.loc5: %.8ce = impl_witness_access constants.%Destroy.impl_witness.9c2, element0 [concrete = constants.%Op.973]
// CHECK:STDOUT:   %bound_method.loc5: <bound method> = bound_method %tuple.var, %impl.elem0.loc5
// CHECK:STDOUT:   %.loc5_27.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_32.1: %tuple.type.b6b = tuple_literal (%.loc5_27.1, %.loc5_31.1)
// CHECK:STDOUT:   %tuple.elem0.loc5_32: ref %empty_struct_type = tuple_access %tuple.var, element0
// CHECK:STDOUT:   %.loc5_27.2: init %empty_struct_type = struct_init () to %tuple.elem0.loc5_32 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_32.2: init %empty_struct_type = converted %.loc5_27.1, %.loc5_27.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc5_32: ref %empty_struct_type = tuple_access %tuple.var, element1
// CHECK:STDOUT:   %.loc5_31.2: init %empty_struct_type = struct_init () to %tuple.elem1.loc5_32 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_32.3: init %empty_struct_type = converted %.loc5_31.1, %.loc5_31.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_32.4: init %tuple.type.b6b = tuple_init (%.loc5_32.2, %.loc5_32.3) to %tuple.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_3.2: init %tuple.type.b6b = converted %.loc5_32.1, %.loc5_32.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %tuple.var, %.loc5_3.2
// CHECK:STDOUT:   %.loc5_21.1: type = splice_block %.loc5_21.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:     %.loc5_16: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_20: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_21.2: %tuple.type.b6b = tuple_literal (%.loc5_16, %.loc5_20)
// CHECK:STDOUT:     %.loc5_21.3: type = converted %.loc5_16, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_21.4: type = converted %.loc5_20, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_21.5: type = converted %.loc5_21.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.loc5_7: ref %tuple.type.b6b = bind_name tuple, %tuple.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc6_20: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:     %.loc6_3.1: %tuple.type.b6b = var_pattern %.loc6_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %impl.elem0.loc6: %.8ce = impl_witness_access constants.%Destroy.impl_witness.9c2, element0 [concrete = constants.%Op.973]
// CHECK:STDOUT:   %bound_method.loc6: <bound method> = bound_method %.var, %impl.elem0.loc6
// CHECK:STDOUT:   %tuple.ref: ref %tuple.type.b6b = name_ref tuple, %tuple.loc5_7
// CHECK:STDOUT:   %tuple.elem0.loc6_24.1: ref %empty_struct_type = tuple_access %tuple.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc6_24.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc6_24.1: init %empty_struct_type = struct_init () to %tuple.elem0.loc6_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_24.2: init %empty_struct_type = converted %tuple.elem0.loc6_24.1, %.loc6_24.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc6_24.1: ref %empty_struct_type = tuple_access %tuple.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc6_24.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc6_24.3: init %empty_struct_type = struct_init () to %tuple.elem1.loc6_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_24.4: init %empty_struct_type = converted %tuple.elem1.loc6_24.1, %.loc6_24.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_24.5: init %tuple.type.b6b = tuple_init (%.loc6_24.2, %.loc6_24.4) to %.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_3.2: init %tuple.type.b6b = converted %tuple.ref, %.loc6_24.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc6_3.2
// CHECK:STDOUT:   %tuple.elem0.loc6_3.1: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc6_3.1: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %tuple.elem0.loc6_3.1
// CHECK:STDOUT:   %.loc6_19.1: type = splice_block %.loc6_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc6_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_19.3: type = converted %.loc6_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = bind_name y, %tuple.elem1.loc6_3.1
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Op.2(constants.%tuple.type.b6b) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.var, %specific_fn.loc6
// CHECK:STDOUT:   %tuple.elem0.loc6_3.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %empty_struct.loc6_3.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_3.3: %empty_struct_type = converted %tuple.elem0.loc6_3.2, %empty_struct.loc6_3.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc6_3.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %empty_struct.loc6_3.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_3.4: %empty_struct_type = converted %tuple.elem1.loc6_3.2, %empty_struct.loc6_3.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc6: %tuple.type.b6b = tuple_value (%.loc6_3.3, %.loc6_3.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_3.5: %tuple.type.b6b = converted %.var, %tuple.loc6 [concrete = constants.%tuple]
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc6_3.5)
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Op.2(constants.%tuple.type.b6b) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %tuple.var, %specific_fn.loc5
// CHECK:STDOUT:   %tuple.elem0.loc5_3: ref %empty_struct_type = tuple_access %tuple.var, element0
// CHECK:STDOUT:   %empty_struct.loc5_3.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_3.3: %empty_struct_type = converted %tuple.elem0.loc5_3, %empty_struct.loc5_3.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc5_3: ref %empty_struct_type = tuple_access %tuple.var, element1
// CHECK:STDOUT:   %empty_struct.loc5_3.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_3.4: %empty_struct_type = converted %tuple.elem1.loc5_3, %empty_struct.loc5_3.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc5_3: %tuple.type.b6b = tuple_value (%.loc5_3.3, %.loc5_3.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_3.5: %tuple.type.b6b = converted %tuple.var, %tuple.loc5_3 [concrete = constants.%tuple]
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc5_3.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.150: %Destroy.type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %tuple.patt: %tuple.type.b6b = binding_pattern tuple
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_27: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_31: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_32.1: %tuple.type.b6b = tuple_literal (%.loc10_27, %.loc10_31)
// CHECK:STDOUT:   %.loc10_21.1: type = splice_block %.loc10_21.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:     %.loc10_16: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_20: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_21.2: %tuple.type.b6b = tuple_literal (%.loc10_16, %.loc10_20)
// CHECK:STDOUT:     %.loc10_21.3: type = converted %.loc10_16, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_21.4: type = converted %.loc10_20, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_21.5: type = converted %.loc10_21.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc10_27: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_32.2: %empty_struct_type = converted %.loc10_27, %empty_struct.loc10_27 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %empty_struct.loc10_31: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_32.3: %empty_struct_type = converted %.loc10_31, %empty_struct.loc10_31 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc10_32: %tuple.type.b6b = tuple_value (%.loc10_32.2, %.loc10_32.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc10_32.4: %tuple.type.b6b = converted %.loc10_32.1, %tuple.loc10_32 [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.loc10_7: %tuple.type.b6b = bind_name tuple, %.loc10_32.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc11_20: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:     %.loc11_3.1: %tuple.type.b6b = var_pattern %.loc11_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %impl.elem0: %.8ce = impl_witness_access constants.%Destroy.impl_witness.9c2, element0 [concrete = constants.%Op.973]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %.var, %impl.elem0
// CHECK:STDOUT:   %tuple.ref: %tuple.type.b6b = name_ref tuple, %tuple.loc10_7
// CHECK:STDOUT:   %tuple.elem0.loc11_24.1: %empty_struct_type = tuple_access %tuple.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_24.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc11_24.1: init %empty_struct_type = struct_init () to %tuple.elem0.loc11_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_24.2: init %empty_struct_type = converted %tuple.elem0.loc11_24.1, %.loc11_24.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc11_24.1: %empty_struct_type = tuple_access %tuple.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc11_24.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc11_24.3: init %empty_struct_type = struct_init () to %tuple.elem1.loc11_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_24.4: init %empty_struct_type = converted %tuple.elem1.loc11_24.1, %.loc11_24.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_24.5: init %tuple.type.b6b = tuple_init (%.loc11_24.2, %.loc11_24.4) to %.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_3.2: init %tuple.type.b6b = converted %tuple.ref, %.loc11_24.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc11_3.2
// CHECK:STDOUT:   %tuple.elem0.loc11_3.1: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc11_3.1: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc11_12.1: type = splice_block %.loc11_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc11_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc11_12.3: type = converted %.loc11_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %tuple.elem0.loc11_3.1
// CHECK:STDOUT:   %.loc11_19.1: type = splice_block %.loc11_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc11_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc11_19.3: type = converted %.loc11_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = bind_name y, %tuple.elem1.loc11_3.1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(constants.%tuple.type.b6b) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.var, %specific_fn
// CHECK:STDOUT:   %tuple.elem0.loc11_3.2: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %empty_struct.loc11_3.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_3.3: %empty_struct_type = converted %tuple.elem0.loc11_3.2, %empty_struct.loc11_3.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc11_3.2: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %empty_struct.loc11_3.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_3.4: %empty_struct_type = converted %tuple.elem1.loc11_3.2, %empty_struct.loc11_3.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc11: %tuple.type.b6b = tuple_value (%.loc11_3.3, %.loc11_3.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_3.5: %tuple.type.b6b = converted %.var, %tuple.loc11 [concrete = constants.%tuple]
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.1(%.loc11_3.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeTuple() -> %return.param: %tuple.type.b6b;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc17_20: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:     %.loc17_3.1: %tuple.type.b6b = var_pattern %.loc17_20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type.b6b = var <none>
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.8ce = impl_witness_access constants.%Destroy.impl_witness.9c2, element0 [concrete = constants.%Op.973]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %.var, %impl.elem0.loc17_3
// CHECK:STDOUT:   %MakeTuple.ref: %MakeTuple.type = name_ref MakeTuple, file.%MakeTuple.decl [concrete = constants.%MakeTuple]
// CHECK:STDOUT:   %.loc17_3.2: ref %tuple.type.b6b = splice_block %.var {}
// CHECK:STDOUT:   %impl.elem0.loc17_34: %.8ce = impl_witness_access constants.%Destroy.impl_witness.9c2, element0 [concrete = constants.%Op.973]
// CHECK:STDOUT:   %bound_method.loc17_34: <bound method> = bound_method %.loc17_3.2, %impl.elem0.loc17_34
// CHECK:STDOUT:   %MakeTuple.call: init %tuple.type.b6b = call %MakeTuple.ref() to %.loc17_3.2
// CHECK:STDOUT:   assign %.var, %MakeTuple.call
// CHECK:STDOUT:   %tuple.elem0.loc17_3.1: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc17_3.1: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc17_12.1: type = splice_block %.loc17_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc17_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_12.3: type = converted %.loc17_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %tuple.elem0.loc17_3.1
// CHECK:STDOUT:   %.loc17_19.1: type = splice_block %.loc17_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc17_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_19.3: type = converted %.loc17_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = bind_name y, %tuple.elem1.loc17_3.1
// CHECK:STDOUT:   %specific_fn.loc17_34: <specific function> = specific_function %impl.elem0.loc17_34, @Op.2(constants.%tuple.type.b6b) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc17_3.2, %specific_fn.loc17_34
// CHECK:STDOUT:   %tuple.elem0.loc17_3.2: ref %empty_struct_type = tuple_access %.loc17_3.2, element0
// CHECK:STDOUT:   %empty_struct.loc17_3.1: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_3.3: %empty_struct_type = converted %tuple.elem0.loc17_3.2, %empty_struct.loc17_3.1 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc17_3.2: ref %empty_struct_type = tuple_access %.loc17_3.2, element1
// CHECK:STDOUT:   %empty_struct.loc17_3.2: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_3.4: %empty_struct_type = converted %tuple.elem1.loc17_3.2, %empty_struct.loc17_3.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc17_3.1: %tuple.type.b6b = tuple_value (%.loc17_3.3, %.loc17_3.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc17_3.5: %tuple.type.b6b = converted %.loc17_3.2, %tuple.loc17_3.1 [concrete = constants.%tuple]
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc17_3.5)
// CHECK:STDOUT:   %specific_fn.loc17_3: <specific function> = specific_function %impl.elem0.loc17_3, @Op.2(constants.%tuple.type.b6b) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.var, %specific_fn.loc17_3
// CHECK:STDOUT:   %tuple.elem0.loc17_3.3: ref %empty_struct_type = tuple_access %.var, element0
// CHECK:STDOUT:   %empty_struct.loc17_3.3: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_3.6: %empty_struct_type = converted %tuple.elem0.loc17_3.3, %empty_struct.loc17_3.3 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.elem1.loc17_3.3: ref %empty_struct_type = tuple_access %.var, element1
// CHECK:STDOUT:   %empty_struct.loc17_3.4: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_3.7: %empty_struct_type = converted %tuple.elem1.loc17_3.3, %empty_struct.loc17_3.4 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %tuple.loc17_3.2: %tuple.type.b6b = tuple_value (%.loc17_3.6, %.loc17_3.7) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc17_3.8: %tuple.type.b6b = converted %.var, %tuple.loc17_3.2 [concrete = constants.%tuple]
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc17_3.8)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.706
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%tuple.type.b6b) {
// CHECK:STDOUT:   %T => constants.%tuple.type.b6b
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.9c2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.72a
// CHECK:STDOUT:   %Op => constants.%Op.973
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%tuple.type.b6b) {
// CHECK:STDOUT:   %T => constants.%tuple.type.b6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.type.6ca: type = tuple_type (%empty_struct_type, %tuple.type.b6b) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %z.patt: %empty_struct_type = binding_pattern z
// CHECK:STDOUT:     %.loc5_28: %tuple.type.b6b = tuple_pattern (%y.patt, %z.patt)
// CHECK:STDOUT:     %.loc5_29: %tuple.type.6ca = tuple_pattern (%x.patt, %.loc5_28)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_35.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_40.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_44.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_45: %tuple.type.b6b = tuple_literal (%.loc5_40.1, %.loc5_44.1)
// CHECK:STDOUT:   %.loc5_46: %tuple.type.6ca = tuple_literal (%.loc5_35.1, %.loc5_45)
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_12.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_35: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_35.2: %empty_struct_type = converted %.loc5_35.1, %empty_struct.loc5_35 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = bind_name x, %.loc5_35.2
// CHECK:STDOUT:   %.loc5_20.1: type = splice_block %.loc5_20.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_20.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_20.3: type = converted %.loc5_20.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_40: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_40.2: %empty_struct_type = converted %.loc5_40.1, %empty_struct.loc5_40 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = bind_name y, %.loc5_40.2
// CHECK:STDOUT:   %.loc5_27.1: type = splice_block %.loc5_27.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc5_27.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_27.3: type = converted %.loc5_27.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc5_44: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_44.2: %empty_struct_type = converted %.loc5_44.1, %empty_struct.loc5_44 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %z: %empty_struct_type = bind_name z, %.loc5_44.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc4_18: %tuple.type = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc4_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc4_24: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc4_24: %empty_struct_type = converted @__global_init.%.loc4_24, %empty_struct.loc4_24 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %x: %empty_struct_type = bind_name x, %.loc4_24
// CHECK:STDOUT:   %.loc4_17.1: type = splice_block %.loc4_17.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc4_17.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_17.3: type = converted %.loc4_17.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct.loc4_28: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc4_28: %empty_struct_type = converted @__global_init.%.loc4_28, %empty_struct.loc4_28 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %y: %empty_struct_type = bind_name y, %.loc4_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_29: %tuple.type = tuple_literal (%.loc4_24, %.loc4_28)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .x = <unexpected>.inst25.loc9_8
// CHECK:STDOUT:   .y = <unexpected>.inst29.loc9_15
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_initializer_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.type.572: type = tuple_type (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = <unexpected>.inst21.loc8_6
// CHECK:STDOUT:     .y = <unexpected>.inst25.loc8_13
// CHECK:STDOUT:     .a = <unexpected>.inst35.loc14_6
// CHECK:STDOUT:     .b = <unexpected>.inst39.loc14_13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %y.patt: %empty_struct_type = binding_pattern y
// CHECK:STDOUT:     %.loc8: %tuple.type.b6b = tuple_pattern (%x.patt, %y.patt)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_struct_type = binding_pattern a
// CHECK:STDOUT:     %b.patt: %empty_struct_type = binding_pattern b
// CHECK:STDOUT:     %.loc14: %tuple.type.b6b = tuple_pattern (%a.patt, %b.patt)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_26: %tuple.type.572 = tuple_literal (%.loc8_24)
// CHECK:STDOUT:   %.loc14_24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_33: %tuple.type.8d4 = tuple_literal (%.loc14_24, %.loc14_28, %.loc14_32)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Op.type.439: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.417: %Op.type.439 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, @Destroy.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness file.%Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.b22: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.d6d: %Op.type.b22 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T, (%Destroy.impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Destroy = %Destroy.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.decl: type = interface_decl @Destroy [concrete = constants.%Destroy.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %Destroy.ref: type = name_ref Destroy, file.%Destroy.decl [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@impl.%Op.decl), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Destroy {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: %Op.type.439 = fn_decl @Op.1 [concrete = constants.%Op.417] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_15.1: type = splice_block %.loc9_15.2 [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Destroy.type = name_ref Self, @Destroy.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_15.2: type = converted %Self.ref, %Self.as_type.loc9_15.2 [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness file.%Destroy.impl_witness_table, @impl(%T.loc11_14.2) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc11_14.2) [symbolic = %Op.type (constants.%Op.type.b22)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.b22) = struct_value () [symbolic = %Op (constants.%Op.d6d)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc11_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %Destroy.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.b22) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.d6d)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Destroy.%Self: %Destroy.type) {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.2.%T (%T)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Destroy.facet) {
// CHECK:STDOUT:   %Self => constants.%Destroy.facet
// CHECK:STDOUT:   %Self.as_type.loc9_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
