// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/extend_name_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/extend_name_lookup.carbon

// --- facet_type_extends_and_constraint_extends.carbon
library "[[@TEST_NAME]]";

interface Z { fn ZF(); }
interface Y { fn YF(); }

constraint W {
  extend require impls Z;
  require impls Y;
}

//@dump-sem-ir-begin
fn F(T:! W) {
  // Extends Z.
  T.ZF();
  // Does not extend Y.
  T.(Y.YF)();
}
//@dump-sem-ir-end

// --- fail_name_lookup_in_constraint_does_not_extend.carbon
library "[[@TEST_NAME]]";

interface Z { fn ZF(); }

constraint W {
  require impls Z;
}

fn F(T:! W) {
  // Does not extend Z.
interface Y { fn YF(); }
  // CHECK:STDERR: fail_name_lookup_in_constraint_does_not_extend.carbon:[[@LINE+4]]:3: error: member name `ZF` not found in `W` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   T.ZF();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.ZF();
}

// --- facet_type_does_not_extend_and_constraint_extends.carbon
library "[[@TEST_NAME]]";

interface Z { fn ZF(); }
interface Y { fn YF(); }

constraint W {
  extend require impls Z;
  require impls Y;
}

//@dump-sem-ir-begin
fn F(T:! type where .Self impls W) {
  // T does not extend Z.
  T.(Z.ZF)();
  // T does not extend Y.
  T.(Y.YF)();
}
//@dump-sem-ir-end

// --- fail_name_lookup_in_facet_type_does_not_extend.carbon
library "[[@TEST_NAME]]";

interface Z { fn ZF(); }
interface Y { fn YF(); }

constraint W {
  extend require impls Z;
  require impls Y;
}

fn F(T:! type where .Self impls W) {
  // T does not extend Z.
  // CHECK:STDERR: fail_name_lookup_in_facet_type_does_not_extend.carbon:[[@LINE+4]]:3: error: member name `ZF` not found [MemberNameNotFound]
  // CHECK:STDERR:   T.ZF();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.ZF();
  // T does not extend Y.
  // CHECK:STDERR: fail_name_lookup_in_facet_type_does_not_extend.carbon:[[@LINE+4]]:3: error: member name `YF` not found [MemberNameNotFound]
  // CHECK:STDERR:   T.YF();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  T.YF();
}

// CHECK:STDOUT: --- facet_type_extends_and_constraint_extends.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.0cc: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z.WithSelf.ZF.decl [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.2e5: %Y.assoc_type = assoc_entity element0, @Y.WithSelf.%Y.WithSelf.YF.decl [concrete]
// CHECK:STDOUT:   %W.type: type = facet_type <@W> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %W.type [concrete]
// CHECK:STDOUT:   %T: %W.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Z [symbolic]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %T.binding.as_type, (%Z.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Z.WithSelf.ZF.type.ac0: type = fn_type @Z.WithSelf.ZF, @Z.WithSelf(%Z.facet) [symbolic]
// CHECK:STDOUT:   %.ec9: type = fn_type_with_self_type %Z.WithSelf.ZF.type.ac0, %Z.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.e55: %.ec9 = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.950: <specific function> = specific_impl_function %impl.elem0.e55, @Z.WithSelf.ZF(%Z.facet) [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Y [symbolic]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Y.WithSelf.YF.type.288: type = fn_type @Y.WithSelf.YF, @Y.WithSelf(%Y.facet) [symbolic]
// CHECK:STDOUT:   %.c28: type = fn_type_with_self_type %Y.WithSelf.YF.type.288, %Y.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.984: %.c28 = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.317: <specific function> = specific_impl_function %impl.elem0.984, @Y.WithSelf.YF(%Y.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12: type = splice_block %W.ref [concrete = constants.%W.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %W.ref: type = name_ref W, file.%W.decl [concrete = constants.%W.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_7.2: %W.type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc12_7.2: %W.type) {
// CHECK:STDOUT:   %T.loc12_7.1: %W.type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc12_7.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_7.1, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness)]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %T.binding.as_type, (%Z.lookup_impl_witness) [symbolic = %Z.facet (constants.%Z.facet)]
// CHECK:STDOUT:   %Z.WithSelf.ZF.type: type = fn_type @Z.WithSelf.ZF, @Z.WithSelf(%Z.facet) [symbolic = %Z.WithSelf.ZF.type (constants.%Z.WithSelf.ZF.type.ac0)]
// CHECK:STDOUT:   %.loc14_4.2: type = fn_type_with_self_type %Z.WithSelf.ZF.type, %Z.facet [symbolic = %.loc14_4.2 (constants.%.ec9)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @F.%.loc14_4.2 (%.ec9) = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.e55)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @Z.WithSelf.ZF(%Z.facet) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.950)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_7.1, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness)]
// CHECK:STDOUT:   %Y.facet.loc16_4.2: %Y.type = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic = %Y.facet.loc16_4.2 (constants.%Y.facet)]
// CHECK:STDOUT:   %Y.WithSelf.YF.type: type = fn_type @Y.WithSelf.YF, @Y.WithSelf(%Y.facet.loc16_4.2) [symbolic = %Y.WithSelf.YF.type (constants.%Y.WithSelf.YF.type.288)]
// CHECK:STDOUT:   %.loc16_4.2: type = fn_type_with_self_type %Y.WithSelf.YF.type, %Y.facet.loc16_4.2 [symbolic = %.loc16_4.2 (constants.%.c28)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @F.%.loc16_4.2 (%.c28) = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0.984)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @Y.WithSelf.YF(%Y.facet.loc16_4.2) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn.317)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc14: %W.type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc14: type = facet_access_type %T.ref.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc14_4.1: type = converted %T.ref.loc14, %T.as_type.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %ZF.ref: %Z.assoc_type = name_ref ZF, @Z.WithSelf.%assoc0 [concrete = constants.%assoc0.0cc]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @F.%.loc14_4.2 (%.ec9) = impl_witness_access constants.%Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.e55)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @Z.WithSelf.ZF(constants.%Z.facet) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.950)]
// CHECK:STDOUT:     %Z.WithSelf.ZF.call: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     %T.ref.loc16: %W.type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     %YF.ref: %Y.assoc_type = name_ref YF, @Y.WithSelf.%assoc0 [concrete = constants.%assoc0.2e5]
// CHECK:STDOUT:     %T.as_type.loc16: type = facet_access_type %T.ref.loc16 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Y.facet.loc16_4.1: %Y.type = facet_value %T.as_type.loc16, (constants.%Y.lookup_impl_witness) [symbolic = %Y.facet.loc16_4.2 (constants.%Y.facet)]
// CHECK:STDOUT:     %.loc16_4.1: %Y.type = converted %T.ref.loc16, %Y.facet.loc16_4.1 [symbolic = %Y.facet.loc16_4.2 (constants.%Y.facet)]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @F.%.loc16_4.2 (%.c28) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0.984)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_4.1: <specific function> = specific_impl_function %impl.elem0.loc16_4.1, @Y.WithSelf.YF(constants.%Y.facet) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn.317)]
// CHECK:STDOUT:     %Y.WithSelf.YF.call: init %empty_tuple.type = call %specific_impl_fn.loc16_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc12_7.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- facet_type_does_not_extend_and_constraint_extends.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.0cc: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z.WithSelf.ZF.decl [concrete]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Y.assoc_type: type = assoc_entity_type @Y [concrete]
// CHECK:STDOUT:   %assoc0.2e5: %Y.assoc_type = assoc_entity element0, @Y.WithSelf.%Y.WithSelf.YF.decl [concrete]
// CHECK:STDOUT:   %W.type: type = facet_type <@W> [concrete]
// CHECK:STDOUT:   %.Self.16f: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %type_where [concrete]
// CHECK:STDOUT:   %T: %type_where = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Z [symbolic]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %T.binding.as_type, (%Z.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Z.WithSelf.ZF.type.ac0: type = fn_type @Z.WithSelf.ZF, @Z.WithSelf(%Z.facet) [symbolic]
// CHECK:STDOUT:   %.ec9: type = fn_type_with_self_type %Z.WithSelf.ZF.type.ac0, %Z.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.e55: %.ec9 = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.950: <specific function> = specific_impl_function %impl.elem0.e55, @Z.WithSelf.ZF(%Z.facet) [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Y [symbolic]
// CHECK:STDOUT:   %Y.facet: %Y.type = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Y.WithSelf.YF.type.288: type = fn_type @Y.WithSelf.YF, @Y.WithSelf(%Y.facet) [symbolic]
// CHECK:STDOUT:   %.c28: type = fn_type_with_self_type %Y.WithSelf.YF.type.288, %Y.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.984: %.c28 = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.317: <specific function> = specific_impl_function %impl.elem0.984, @Y.WithSelf.YF(%Y.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_15.1: type = splice_block %.loc12_15.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc12_10: type = type_literal type [concrete = type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.16f]
// CHECK:STDOUT:       %W.ref: type = name_ref W, file.%W.decl [concrete = constants.%W.type]
// CHECK:STDOUT:       %.loc12_15.2: type = where_expr %.Self.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_base_facet_type type
// CHECK:STDOUT:         requirement_impls %.Self.ref, %W.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_7.2: %type_where = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc12_7.2: %type_where) {
// CHECK:STDOUT:   %T.loc12_7.1: %type_where = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc12_7.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_7.1, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness)]
// CHECK:STDOUT:   %Z.facet.loc14_4.2: %Z.type = facet_value %T.binding.as_type, (%Z.lookup_impl_witness) [symbolic = %Z.facet.loc14_4.2 (constants.%Z.facet)]
// CHECK:STDOUT:   %Z.WithSelf.ZF.type: type = fn_type @Z.WithSelf.ZF, @Z.WithSelf(%Z.facet.loc14_4.2) [symbolic = %Z.WithSelf.ZF.type (constants.%Z.WithSelf.ZF.type.ac0)]
// CHECK:STDOUT:   %.loc14_4.2: type = fn_type_with_self_type %Z.WithSelf.ZF.type, %Z.facet.loc14_4.2 [symbolic = %.loc14_4.2 (constants.%.ec9)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @F.%.loc14_4.2 (%.ec9) = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.e55)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @Z.WithSelf.ZF(%Z.facet.loc14_4.2) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.950)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_7.1, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness)]
// CHECK:STDOUT:   %Y.facet.loc16_4.2: %Y.type = facet_value %T.binding.as_type, (%Y.lookup_impl_witness) [symbolic = %Y.facet.loc16_4.2 (constants.%Y.facet)]
// CHECK:STDOUT:   %Y.WithSelf.YF.type: type = fn_type @Y.WithSelf.YF, @Y.WithSelf(%Y.facet.loc16_4.2) [symbolic = %Y.WithSelf.YF.type (constants.%Y.WithSelf.YF.type.288)]
// CHECK:STDOUT:   %.loc16_4.2: type = fn_type_with_self_type %Y.WithSelf.YF.type, %Y.facet.loc16_4.2 [symbolic = %.loc16_4.2 (constants.%.c28)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: @F.%.loc16_4.2 (%.c28) = impl_witness_access %Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0.984)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_4.2: <specific function> = specific_impl_function %impl.elem0.loc16_4.2, @Y.WithSelf.YF(%Y.facet.loc16_4.2) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn.317)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc14: %type_where = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %ZF.ref: %Z.assoc_type = name_ref ZF, @Z.WithSelf.%assoc0 [concrete = constants.%assoc0.0cc]
// CHECK:STDOUT:     %T.as_type.loc14: type = facet_access_type %T.ref.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Z.facet.loc14_4.1: %Z.type = facet_value %T.as_type.loc14, (constants.%Z.lookup_impl_witness) [symbolic = %Z.facet.loc14_4.2 (constants.%Z.facet)]
// CHECK:STDOUT:     %.loc14_4.1: %Z.type = converted %T.ref.loc14, %Z.facet.loc14_4.1 [symbolic = %Z.facet.loc14_4.2 (constants.%Z.facet)]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @F.%.loc14_4.2 (%.ec9) = impl_witness_access constants.%Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.e55)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @Z.WithSelf.ZF(constants.%Z.facet) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.950)]
// CHECK:STDOUT:     %Z.WithSelf.ZF.call: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     %T.ref.loc16: %type_where = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T)]
// CHECK:STDOUT:     %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y.type]
// CHECK:STDOUT:     %YF.ref: %Y.assoc_type = name_ref YF, @Y.WithSelf.%assoc0 [concrete = constants.%assoc0.2e5]
// CHECK:STDOUT:     %T.as_type.loc16: type = facet_access_type %T.ref.loc16 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Y.facet.loc16_4.1: %Y.type = facet_value %T.as_type.loc16, (constants.%Y.lookup_impl_witness) [symbolic = %Y.facet.loc16_4.2 (constants.%Y.facet)]
// CHECK:STDOUT:     %.loc16_4.1: %Y.type = converted %T.ref.loc16, %Y.facet.loc16_4.1 [symbolic = %Y.facet.loc16_4.2 (constants.%Y.facet)]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: @F.%.loc16_4.2 (%.c28) = impl_witness_access constants.%Y.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0.984)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_4.1: <specific function> = specific_impl_function %impl.elem0.loc16_4.1, @Y.WithSelf.YF(constants.%Y.facet) [symbolic = %specific_impl_fn.loc16_4.2 (constants.%specific_impl_fn.317)]
// CHECK:STDOUT:     %Y.WithSelf.YF.call: init %empty_tuple.type = call %specific_impl_fn.loc16_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc12_7.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
