// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/pointer.carbon

// ============================================================================
// Pointer as a parameter type
// ============================================================================

// --- pointer_param.h

struct S {};

auto foo(S* _Nonnull) -> void;

// --- import_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  Cpp.foo(&s);
  //@dump-sem-ir-end
}

// ============================================================================
// Double pointer as a parameter type
// ============================================================================

// --- double_pointer_param.h

struct S {};

auto foo(S* _Nonnull * _Nonnull) -> void;

// --- import_double_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  var p: Cpp.S* = &s;
  Cpp.foo(&p);
  //@dump-sem-ir-end
}

// ============================================================================
// Const pointer as a parameter type
// ============================================================================

// --- const_pointer_param.h

struct S {};

auto foo(const S* _Nonnull) -> void;

// --- import_const_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_pointer_param.h";

fn G() -> const Cpp.S;

fn F() {
  //@dump-sem-ir-begin
  var s: const Cpp.S = G();
  Cpp.foo(&s);
  //@dump-sem-ir-end
}

// ============================================================================
// Pointer as a return value
// ============================================================================

// --- pointer_return.h

struct S {};

auto foo() -> S* _Nonnull;

// --- import_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "pointer_return.h";

fn IngestDoublePointer(s: Cpp.S*);

fn F() {
  //@dump-sem-ir-begin
  IngestDoublePointer(Cpp.foo());
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Double pointer as a return value
// ============================================================================

// --- double_pointer_return.h

struct S {};

auto foo() -> S* _Nonnull * _Nonnull;

// --- import_double_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_pointer_return.h";

fn IngestDoublePointer(s: Cpp.S**);

fn F() {
  //@dump-sem-ir-begin
  IngestDoublePointer(Cpp.foo());
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Const pointer as a return value
// ============================================================================

// --- const_pointer_return.h

struct S {};

auto foo() -> const S* _Nonnull;

// --- import_const_pointer_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_pointer_return.h";

fn IngestConstPointer(s: const Cpp.S*);

fn F() {
  //@dump-sem-ir-begin
  IngestConstPointer(Cpp.foo());
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Nullable pointer
// ============================================================================

// --- nullable_pointer_param.h

struct S {};

auto foo(S*) -> void;

// --- fail_todo_import_nullable_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "nullable_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  // CHECK:STDERR: fail_todo_import_nullable_pointer_param.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: S *` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(&s);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_nullable_pointer_param.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.foo(&s);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(&s);
  //@dump-sem-ir-end
}

// ============================================================================
// Deduced pointer type as template argument
// ============================================================================

// --- deduced_any_param.h

struct S {};

template<typename T>
auto foo(T) -> T;

// --- import_deduced_any_param_as_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "deduced_any_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  let p: Cpp.S* = Cpp.foo(&s);
  //@dump-sem-ir-end
}

// --- deduced_pointer_param.h

struct S {};

template<typename T>
auto Direct(T*) -> T*;

template<typename T>
using Wrap = T;

template<typename U>
using AddPointer = Wrap<U*>;

template<typename V>
auto Indirect(V) -> AddPointer<V>;

// --- fail_import_deduced_pointer_param_as_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "deduced_pointer_param.h";

fn F() {
  //@dump-sem-ir-begin
  var s: Cpp.S = {};
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_pointer.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: struct S *` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.Direct(&s);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_pointer.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.Direct(&s);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Direct(&s);

  // CHECK:STDERR: fail_import_deduced_pointer_param_as_pointer.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: return type: AddPointer<struct S>` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.Indirect({} as Cpp.S);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_deduced_pointer_param_as_pointer.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.Indirect({} as Cpp.S);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Indirect({} as Cpp.S);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.cc2: %AggregateT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3.1: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc9: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%addr.loc9)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.692: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.bf9: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.1cb: %ptr.as.Copy.impl.Op.type.bf9 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.7cc: %Copy.type = facet_value %ptr.5c7, (%Copy.impl_witness.692) [concrete]
// CHECK:STDOUT:   %.dce: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.7cc [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.1cb, @ptr.as.Copy.impl.Op(%S) [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @Copy.Op [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dfe: type = ptr_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.type.c2a: type = fn_type @ptr.as.Destroy.impl.Op, @ptr.as.Destroy.impl(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.431: %ptr.as.Destroy.impl.Op.type.c2a = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.cc2: %AggregateT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @Copy.Op [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3.1: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.259 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.5c7 = var %p.var_patt
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc9_19: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %impl.elem0: %.dce = impl_witness_access constants.%Copy.impl_witness.692, element0 [concrete = constants.%ptr.as.Copy.impl.Op.1cb]
// CHECK:STDOUT:   %bound_method.loc9_19.1: <bound method> = bound_method %addr.loc9_19, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%S) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_19.2: <bound method> = bound_method %addr.loc9_19, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.5c7 = call %bound_method.loc9_19.2(%addr.loc9_19)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref.loc9 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.5c7 = bind_name p, %p.var
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %p.ref: ref %ptr.5c7 = name_ref p, %p
// CHECK:STDOUT:   %addr.loc10: %ptr.dfe = addr_of %p.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%addr.loc10)
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.bound: <bound method> = bound_method %p.var, constants.%ptr.as.Destroy.impl.Op.431
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9_3: <bound method> = bound_method %p.var, %ptr.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc9_3: %ptr.dfe = addr_of %p.var
// CHECK:STDOUT:   %ptr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_3(%addr.loc9_3)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %s.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const.e39: type = const_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.9be: type = pattern_type %const.e39 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @F [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ff5: type = ptr_type %const.e39 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %AggregateT: %type_where = bind_symbolic_name AggregateT, 0 [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.190: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%AggregateT) [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.8a0: %AggregateT.as_type.as.Destroy.impl.Op.type.190 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c8b: <witness> = impl_witness imports.%Destroy.impl_witness_table.2d3, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7eb: %Destroy.type = facet_value %S, (%Destroy.impl_witness.c8b) [concrete]
// CHECK:STDOUT:   %const.as.Destroy.impl.Op.type.21f: type = fn_type @const.as.Destroy.impl.Op, @const.as.Destroy.impl(%Destroy.facet.7eb) [concrete]
// CHECK:STDOUT:   %const.as.Destroy.impl.Op.a1c: %const.as.Destroy.impl.Op.type.21f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @F [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d51: @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op.type (%AggregateT.as_type.as.Destroy.impl.Op.type.190) = import_ref Core//prelude/parts/destroy, loc29_29, loaded [symbolic = @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op (constants.%AggregateT.as_type.as.Destroy.impl.Op.8a0)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.2d3 = impl_witness_table (%Core.import_ref.d51), @AggregateT.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.9be = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.9be = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %const.e39 = var %s.var_patt
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc10_3.1: ref %const.e39 = splice_block %s.var {}
// CHECK:STDOUT:   %G.call: init %const.e39 = call %G.ref() to %.loc10_3.1
// CHECK:STDOUT:   assign %s.var, %G.call
// CHECK:STDOUT:   %.loc10_10: type = splice_block %const [concrete = constants.%const.e39] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %const: type = const_type %S.ref [concrete = constants.%const.e39]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %const.e39 = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref: ref %const.e39 = name_ref s, %s
// CHECK:STDOUT:   %addr.loc11: %ptr.ff5 = addr_of %s.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%addr.loc11)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_3.2: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value constants.%S, (constants.%Destroy.impl_witness.c8b) [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%S, %Destroy.facet [concrete = constants.%Destroy.facet.7eb]
// CHECK:STDOUT:   %const.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%const.as.Destroy.impl.Op.a1c
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %const.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc10: %ptr.ff5 = addr_of %s.var
// CHECK:STDOUT:   %const.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc10)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %IngestDoublePointer.type: type = fn_type @IngestDoublePointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestDoublePointer: %IngestDoublePointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @F [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @F [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IngestDoublePointer.ref: %IngestDoublePointer.type = name_ref IngestDoublePointer, file.%IngestDoublePointer.decl [concrete = constants.%IngestDoublePointer]
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc10: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call.loc10: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_31.1: %ptr = value_of_initializer %foo.call.loc10
// CHECK:STDOUT:   %.loc10_31.2: %ptr = converted %foo.call.loc10, %.loc10_31.1
// CHECK:STDOUT:   %IngestDoublePointer.call: init %empty_tuple.type = call %IngestDoublePointer.ref(%.loc10_31.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc11: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call.loc11: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %ptr.dfe: type = ptr_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %IngestDoublePointer.type: type = fn_type @IngestDoublePointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestDoublePointer: %IngestDoublePointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @F [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @F [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IngestDoublePointer.ref: %IngestDoublePointer.type = name_ref IngestDoublePointer, file.%IngestDoublePointer.decl [concrete = constants.%IngestDoublePointer]
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc10: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call.loc10: init %ptr.dfe = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_31.1: %ptr.dfe = value_of_initializer %foo.call.loc10
// CHECK:STDOUT:   %.loc10_31.2: %ptr.dfe = converted %foo.call.loc10, %.loc10_31.1
// CHECK:STDOUT:   %IngestDoublePointer.call: init %empty_tuple.type = call %IngestDoublePointer.ref(%.loc10_31.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc11: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call.loc11: init %ptr.dfe = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_pointer_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %const: type = const_type %S [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %IngestConstPointer.type: type = fn_type @IngestConstPointer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IngestConstPointer: %IngestConstPointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @F [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @F [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IngestConstPointer.ref: %IngestConstPointer.type = name_ref IngestConstPointer, file.%IngestConstPointer.decl [concrete = constants.%IngestConstPointer]
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc10: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call.loc10: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_30.1: %ptr = value_of_initializer %foo.call.loc10
// CHECK:STDOUT:   %.loc10_30.2: %ptr = converted %foo.call.loc10, %.loc10_30.1
// CHECK:STDOUT:   %IngestConstPointer.call: init %empty_tuple.type = call %IngestConstPointer.ref(%.loc10_30.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc11: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.call.loc11: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_nullable_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.cc2: %AggregateT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @Destroy.Op [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3.1: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc16: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_deduced_any_param_as_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.cc2: %AggregateT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3.1: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc9: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %foo.call: init %ptr.5c7 = call imports.%foo.decl(%addr.loc9)
// CHECK:STDOUT:   %.loc9_15: type = splice_block %ptr [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc9_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr: type = ptr_type %S.ref.loc9 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_29.1: %ptr.5c7 = value_of_initializer %foo.call
// CHECK:STDOUT:   %.loc9_29.2: %ptr.5c7 = converted %foo.call, %.loc9_29.1
// CHECK:STDOUT:   %p: %ptr.5c7 = bind_name p, %.loc9_29.2
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%S, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %s.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_deduced_pointer_param_as_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value () [concrete]
// CHECK:STDOUT:   %.97c: type = cpp_overload_set_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %empty_struct.504: %.97c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %.fed: type = cpp_overload_set_type @ptr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_struct.6e8: %.fed = struct_value () [concrete]
// CHECK:STDOUT:   %Indirect.type: type = fn_type @Indirect [concrete]
// CHECK:STDOUT:   %Indirect: %Indirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %S, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.df1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.cc2: %AggregateT.as_type.as.Destroy.impl.Op.type.df1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .Direct = %.e4a
// CHECK:STDOUT:     .Indirect = %.9c4
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %.e4a: %.97c = cpp_overload_set_value @Destroy.Op [concrete = constants.%empty_struct.504]
// CHECK:STDOUT:   %.9c4: %.fed = cpp_overload_set_value @ptr.as.Destroy.impl.Op [concrete = constants.%empty_struct.6e8]
// CHECK:STDOUT:   %Indirect.decl: %Indirect.type = fn_decl @Indirect [concrete = constants.%Indirect] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.7da = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.7da = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref %S = var %s.var_patt
// CHECK:STDOUT:   %.loc8_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_19.2: init %S = class_init (), %s.var [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc8_3.1: init %S = converted %.loc8_19.1, %.loc8_19.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   assign %s.var, %.loc8_3.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %S.ref.loc8 [concrete = constants.%S] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc8: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: ref %S = bind_name s, %s.var
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Direct.ref: %.97c = name_ref Direct, imports.%.e4a [concrete = constants.%empty_struct.504]
// CHECK:STDOUT:   %s.ref: ref %S = name_ref s, %s
// CHECK:STDOUT:   %addr.loc16: %ptr.5c7 = addr_of %s.ref
// CHECK:STDOUT:   %Cpp.ref.loc25_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Indirect.ref: %.fed = name_ref Indirect, imports.%.9c4 [concrete = constants.%empty_struct.6e8]
// CHECK:STDOUT:   %.loc25_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc25_22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc25: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %.loc25_17.2: ref %S = temporary_storage
// CHECK:STDOUT:   %.loc25_17.3: init %S = class_init (), %.loc25_17.2 [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc25_17.4: ref %S = temporary %.loc25_17.2, %.loc25_17.3
// CHECK:STDOUT:   %.loc25_19.1: ref %S = converted %.loc25_17.1, %.loc25_17.4
// CHECK:STDOUT:   %.loc25_19.2: %S = bind_value %.loc25_19.1
// CHECK:STDOUT:   %Indirect.call: init <error> = call imports.%Indirect.decl(%.loc25_19.2)
// CHECK:STDOUT:   %facet_value.loc25: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc25_17.5: %type_where = converted constants.%S, %facet_value.loc25 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc25: <bound method> = bound_method %.loc25_17.4, constants.%AggregateT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc25: <bound method> = bound_method %.loc25_17.4, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc25: %ptr.5c7 = addr_of %.loc25_17.4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc25: init %empty_tuple.type = call %bound_method.loc25(%addr.loc25)
// CHECK:STDOUT:   %facet_value.loc8: %type_where = facet_value constants.%S, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %type_where = converted constants.%S, %facet_value.loc8 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %s.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.cc2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %s.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8: %ptr.5c7 = addr_of %s.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
