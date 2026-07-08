// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/export/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/export/generic.carbon

// --- generic_type_impls_interface.carbon
library "[[@TEST_NAME]]";
import Cpp;

//@dump-sem-ir-begin
interface I {
  fn Doit(self);
}

class A {
  impl as I {
    fn Doit(unused self) {}
  }
}
class B {
  impl as I {
    fn Doit(unused self) {}
  }
}

fn F[T:! I](t: T) {
  t.Doit();
}

inline Cpp '''
void G() {
  Carbon::A a;
  Carbon::B b;
  Carbon::F(a);
  Carbon::F(b);
}
''';
//@dump-sem-ir-end

// --- fail_todo_non_type_generic.carbon
library "[[@TEST_NAME]]";
import Cpp;

// CHECK:STDERR: fail_todo_non_type_generic.carbon:[[@LINE+4]]:1: error: semantics TODO: `binding maps to a non-type template parameter` [SemanticsTodo]
// CHECK:STDERR: fn F[unused N:! i32]() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F[unused N:! i32]() {}

inline Cpp '''
void G() {
  // CHECK:STDERR: fail_todo_non_type_generic.carbon:[[@LINE+8]]:11: error: no member named 'F' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    20 |   Carbon::F<3>();
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_non_type_generic.carbon:[[@LINE+4]]:16: error: expected expression [CppInteropParseError]
  // CHECK:STDERR:    20 |   Carbon::F<3>();
  // CHECK:STDERR:       |                ^
  // CHECK:STDERR:
  Carbon::F<3>();
}
''';

// --- fail_todo_generic_pointer.carbon
library "[[@TEST_NAME]]";
import Cpp;

fn F[T:! type](unused t: T*) {}

inline Cpp '''
void G() {
  int x = 0;
  int* _Nonnull p = &x;
  // CHECK:STDERR: fail_todo_generic_pointer.carbon:[[@LINE+7]]:3: error: no matching function for call to 'F' [CppInteropParseError]
  // CHECK:STDERR:    17 |   Carbon::F(p);
  // CHECK:STDERR:       |   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_pointer.carbon:[[@LINE-9]]:30: note: candidate template ignored: deduced type 'T * _Nonnull' of 1st parameter does not match adjusted type 'int * _Nonnull' of argument [with T = int] [CppInteropParseNote]
  // CHECK:STDERR:     4 | fn F[T:! type](unused t: T*) {}
  // CHECK:STDERR:       |                              ^
  // CHECK:STDERR:
  Carbon::F(p);
}
''';

// --- fail_todo_enclosing_generic.carbon
library "[[@TEST_NAME]]";
import Cpp;

class A(T:! type) {
  fn F[U:! type](x: T, y: U);
}
alias B = A(i32);

inline Cpp '''
void f() {
  // TODO: this currently fails because `B` can't be imported, but the intent
  // is to test a generic function within an enclosing generic scope.
  //
  // CHECK:STDERR: fail_todo_enclosing_generic.carbon:[[@LINE+8]]:11: error: semantics TODO: `interop with unsupported type` [SemanticsTodo]
  // CHECK:STDERR:   Carbon::B b;
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_enclosing_generic.carbon:[[@LINE+4]]:11: error: no type named 'B' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    22 |   Carbon::B b;
  // CHECK:STDERR:       |   ~~~~~~~~^
  // CHECK:STDERR:
  Carbon::B b;
  b.F(1, 2);
}
''';

// --- fail_generic_unsupported_type.carbon
library "[[@TEST_NAME]]";
import Cpp;

// CHECK:STDERR: fail_generic_unsupported_type.carbon:[[@LINE+4]]:1: error: semantics TODO: `failed to import C++ type` [SemanticsTodo]
// CHECK:STDERR: fn F[T:! type](unused t: T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F[T:! type](unused t: T) {}

inline Cpp '''
void G() {
  volatile int x;
  // CHECK:STDERR: fail_generic_unsupported_type.carbon:[[@LINE+7]]:3: error: no matching function for call to 'F' [CppInteropParseError]
  // CHECK:STDERR:    20 |   Carbon::F(x);
  // CHECK:STDERR:       |   ^~~~~~~~~
  // CHECK:STDERR: fail_generic_unsupported_type.carbon:[[@LINE-8]]:29: note: candidate template ignored: substitution failure [with T = volatile int] [CppInteropParseNote]
  // CHECK:STDERR:     8 | fn F[T:! type](unused t: T) {}
  // CHECK:STDERR:       |                             ^
  // CHECK:STDERR:
  Carbon::F(x);
}
''';

// CHECK:STDOUT: --- generic_type_impls_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %pattern_type.c60: type = pattern_type %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %self.param_patt.a33: %pattern_type.c60 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.696: %pattern_type.c60 = at_binding_pattern self, %self.param_patt.a33 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.26f: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.11f: %I.WithSelf.Doit.type.26f = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.628: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Doit.decl [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %I.impl_witness.6ad: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.312: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %A.as.I.impl.Doit.type: type = fn_type @A.as.I.impl.Doit [concrete]
// CHECK:STDOUT:   %A.as.I.impl.Doit: %A.as.I.impl.Doit.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.64b: %I.type = facet_value %A, (%I.impl_witness.6ad) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.5b3: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%I.facet.64b) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.dab: %I.WithSelf.Doit.type.5b3 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %I.impl_witness.5b8: <witness> = impl_witness @B.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %self.param_patt.f1f: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4eb: %pattern_type.e39 = at_binding_pattern self, %self.param_patt.f1f [concrete]
// CHECK:STDOUT:   %B.as.I.impl.Doit.type: type = fn_type @B.as.I.impl.Doit [concrete]
// CHECK:STDOUT:   %B.as.I.impl.Doit: %B.as.I.impl.Doit.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.9f8: %I.type = facet_value %B, (%I.impl_witness.5b8) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.2ce: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%I.facet.9f8) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.450: %I.WithSelf.Doit.type.2ce = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt.5db: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.030: %pattern_type.b3a = at_binding_pattern t, %t.param_patt.5db [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.860: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.45d: %I.WithSelf.Doit.type.860 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.8a9: type = fn_type_with_self_type %I.WithSelf.Doit.type.860, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.8a9 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.58e: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.300: %pattern_type.b3a = at_binding_pattern self, %self.param_patt.58e [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Doit(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc9_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e5: %Destroy.type = facet_value %A, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.09e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6e5) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.09e, %Destroy.facet.6e5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f70: %Destroy.type = facet_value %B, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.cda: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f70) [concrete]
// CHECK:STDOUT:   %.3d9: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.cda, %Destroy.facet.f70 [concrete]
// CHECK:STDOUT:   %t.param_patt.169: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.ee8: %pattern_type.9ef = at_binding_pattern t, %t.param_patt.169 [concrete]
// CHECK:STDOUT:   %F.specific_fn.f33: <specific function> = specific_function %F, @F(%I.facet.64b) [concrete]
// CHECK:STDOUT:   %t.param_patt.267: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.062: %pattern_type.e39 = at_binding_pattern t, %t.param_patt.267 [concrete]
// CHECK:STDOUT:   %F.specific_fn.db2: <specific function> = specific_function %F, @F(%I.facet.9f8) [concrete]
// CHECK:STDOUT:   %.660: type = fn_type_with_self_type %I.WithSelf.Doit.type.5b3, %I.facet.64b [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %I.WithSelf.Doit.type.2ce, %I.facet.9f8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc20_7.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc20_14.1: @F.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %t.param_patt.loc20_14.2 (constants.%t.param_patt.5db)]
// CHECK:STDOUT:     %t.patt.loc20_14.1: @F.%pattern_type (%pattern_type.b3a) = at_binding_pattern t, %t.param_patt.loc20_14.1 [symbolic = %t.patt.loc20_14.2 (constants.%t.patt.030)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20_10: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20_7.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc20_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc20_16.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc20_16.1: type = splice_block %.loc20_16.2 [symbolic = %T.as_type.loc20_16.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc20_7.2 [symbolic = %T.loc20_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc20_16.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc20_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc20_16.2: type = converted %T.ref, %T.as_type.loc20_16.2 [symbolic = %T.as_type.loc20_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc20_16.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   inline_cpp "void G() {\n  Carbon::A a;\n  Carbon::B b;\n  Carbon::F(a);\n  Carbon::F(b);\n}\n"
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.Doit.decl: @I.WithSelf.%I.WithSelf.Doit.type (%I.WithSelf.Doit.type.26f) = fn_decl @I.WithSelf.Doit [symbolic = @I.WithSelf.%I.WithSelf.Doit (constants.%I.WithSelf.Doit.11f)] {
// CHECK:STDOUT:     %self.param_patt.loc6_11.1: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc6_11.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:     %self.patt.loc6_11.1: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = at_binding_pattern self, %self.param_patt.loc6_11.1 [symbolic = %self.patt.loc6_11.2 (constants.%self.patt.696)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84) = value_param call_param0
// CHECK:STDOUT:     %.loc6_11.1: type = splice_block %.loc6_11.2 [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:       %Self.as_type.loc6_11.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:       %.loc6_11.2: type = converted %Self.ref, %Self.as_type.loc6_11.2 [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.Doit.decl [concrete = constants.%assoc0.628]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Doit = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.Doit.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %A.as.I.impl.Doit.decl: %A.as.I.impl.Doit.type = fn_decl @A.as.I.impl.Doit [concrete = constants.%A.as.I.impl.Doit] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%self.param_patt.04e]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.312]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%A.as.I.impl.Doit.decl), @A.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.6ad]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Doit = %A.as.I.impl.Doit.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %B.as.I.impl.Doit.decl: %B.as.I.impl.Doit.type = fn_decl @B.as.I.impl.Doit [concrete = constants.%B.as.I.impl.Doit] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = value_param_pattern [concrete = constants.%self.param_patt.f1f]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4eb]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%B.as.I.impl.Doit.decl), @B.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.5b8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Doit = %B.as.I.impl.Doit.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   impl_decl @A.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   impl_decl @B.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.Doit(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:   %Self.as_type.loc6_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc6_11.1 [symbolic = %pattern_type (constants.%pattern_type.c60)]
// CHECK:STDOUT:   %self.param_patt.loc6_11.2: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc6_11.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:   %self.patt.loc6_11.2: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = at_binding_pattern self, %self.param_patt.loc6_11.2 [symbolic = %self.patt.loc6_11.2 (constants.%self.patt.696)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.I.impl.Doit(%self.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.I.impl.Doit(%self.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc20_7.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc20_7.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc20_7.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc20_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc20_16.1: type = facet_access_type %T.loc20_7.1 [symbolic = %T.as_type.loc20_16.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc20_16.1 [symbolic = %pattern_type (constants.%pattern_type.b3a)]
// CHECK:STDOUT:   %t.param_patt.loc20_14.2: @F.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %t.param_patt.loc20_14.2 (constants.%t.param_patt.5db)]
// CHECK:STDOUT:   %t.patt.loc20_14.2: @F.%pattern_type (%pattern_type.b3a) = at_binding_pattern t, %t.param_patt.loc20_14.2 [symbolic = %t.patt.loc20_14.2 (constants.%t.patt.030)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc20_16.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.WithSelf.Doit.type: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%T.loc20_7.1) [symbolic = %I.WithSelf.Doit.type (constants.%I.WithSelf.Doit.type.860)]
// CHECK:STDOUT:   %.loc21_4: type = fn_type_with_self_type %I.WithSelf.Doit.type, %T.loc20_7.1 [symbolic = %.loc21_4 (constants.%.8a9)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc20_7.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc21_4.2: @F.%.loc21_4 (%.8a9) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc21_4.2: <specific function> = specific_impl_function %impl.elem0.loc21_4.2, @I.WithSelf.Doit(%T.loc20_7.1) [symbolic = %specific_impl_fn.loc21_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc20_16.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc20_16.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %Doit.ref: %I.assoc_type = name_ref Doit, @I.WithSelf.%assoc0 [concrete = constants.%assoc0.628]
// CHECK:STDOUT:     %impl.elem0.loc21_4.1: @F.%.loc21_4 (%.8a9) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc21_4: <bound method> = bound_method %t.ref, %impl.elem0.loc21_4.1
// CHECK:STDOUT:     %.loc21_10.1: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc20_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc21_10.2: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc20_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc21_4.1: <specific function> = specific_impl_function %impl.elem0.loc21_4.1, @I.WithSelf.Doit(constants.%T) [symbolic = %specific_impl_fn.loc21_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc21_10: <bound method> = bound_method %t.ref, %specific_impl_fn.loc21_4.1
// CHECK:STDOUT:     %I.WithSelf.Doit.call: init %empty_tuple.type = call %bound_method.loc21_10(%t.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.__destroy_thunk(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %impl.elem0: %.dca = impl_witness_access constants.%custom_witness.df9cc1.2, element0 [concrete = constants.%Destroy.Op.1a2547.2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.param, %impl.elem0
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %bound_method(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_9.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.__destroy_thunk(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %impl.elem0: %.3d9 = impl_witness_access constants.%custom_witness.df9cc1.3, element0 [concrete = constants.%Destroy.Op.1a2547.3]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.param, %impl.elem0
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %bound_method(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F__carbon_thunkinst64000035(%_.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F__carbon_thunkinst64000035.ref: %F.type = name_ref F__carbon_thunkinst64000035, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %I.facet.loc20_19.1: %I.type = facet_value constants.%A, (constants.%I.impl_witness.6ad) [concrete = constants.%I.facet.64b]
// CHECK:STDOUT:   %.loc20_19.1: %I.type = converted constants.%A, %I.facet.loc20_19.1 [concrete = constants.%I.facet.64b]
// CHECK:STDOUT:   %I.facet.loc20_19.2: %I.type = facet_value constants.%A, (constants.%I.impl_witness.6ad) [concrete = constants.%I.facet.64b]
// CHECK:STDOUT:   %.loc20_19.2: %I.type = converted constants.%A, %I.facet.loc20_19.2 [concrete = constants.%I.facet.64b]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F__carbon_thunkinst64000035.ref, @F(constants.%I.facet.64b) [concrete = constants.%F.specific_fn.f33]
// CHECK:STDOUT:   %.loc20_19.3: %A = acquire_value %_.param
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc20_19.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F__carbon_thunkinst64000050(%_.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F__carbon_thunkinst64000050.ref: %F.type = name_ref F__carbon_thunkinst64000050, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %I.facet.loc20_19.1: %I.type = facet_value constants.%B, (constants.%I.impl_witness.5b8) [concrete = constants.%I.facet.9f8]
// CHECK:STDOUT:   %.loc20_19.1: %I.type = converted constants.%B, %I.facet.loc20_19.1 [concrete = constants.%I.facet.9f8]
// CHECK:STDOUT:   %I.facet.loc20_19.2: %I.type = facet_value constants.%B, (constants.%I.impl_witness.5b8) [concrete = constants.%I.facet.9f8]
// CHECK:STDOUT:   %.loc20_19.2: %I.type = converted constants.%B, %I.facet.loc20_19.2 [concrete = constants.%I.facet.9f8]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F__carbon_thunkinst64000050.ref, @F(constants.%I.facet.9f8) [concrete = constants.%F.specific_fn.db2]
// CHECK:STDOUT:   %.loc20_19.3: %B = acquire_value %_.param
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc20_19.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.26f
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.11f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%Self.37e) {
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %Self.as_type.loc6_11.1 => constants.%Self.as_type.c84
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c60
// CHECK:STDOUT:   %self.param_patt.loc6_11.2 => constants.%self.param_patt.a33
// CHECK:STDOUT:   %self.patt.loc6_11.2 => constants.%self.patt.696
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.64b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.64b
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.5b3
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.dab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%I.facet.64b) {
// CHECK:STDOUT:   %Self => constants.%I.facet.64b
// CHECK:STDOUT:   %Self.as_type.loc6_11.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ef
// CHECK:STDOUT:   %self.param_patt.loc6_11.2 => constants.%self.param_patt.04e
// CHECK:STDOUT:   %self.patt.loc6_11.2 => constants.%self.patt.312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.9f8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.9f8
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.2ce
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.450
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%I.facet.9f8) {
// CHECK:STDOUT:   %Self => constants.%I.facet.9f8
// CHECK:STDOUT:   %Self.as_type.loc6_11.1 => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e39
// CHECK:STDOUT:   %self.param_patt.loc6_11.2 => constants.%self.param_patt.f1f
// CHECK:STDOUT:   %self.patt.loc6_11.2 => constants.%self.patt.4eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc20_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc20_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc20_16.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %t.param_patt.loc20_14.2 => constants.%t.param_patt.5db
// CHECK:STDOUT:   %t.patt.loc20_14.2 => constants.%t.patt.030
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.860
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.45d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type.loc6_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %self.param_patt.loc6_11.2 => constants.%self.param_patt.58e
// CHECK:STDOUT:   %self.patt.loc6_11.2 => constants.%self.patt.300
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%I.facet.64b) {
// CHECK:STDOUT:   %T.patt.loc20_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc20_7.1 => constants.%I.facet.64b
// CHECK:STDOUT:   %T.as_type.loc20_16.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ef
// CHECK:STDOUT:   %t.param_patt.loc20_14.2 => constants.%t.param_patt.169
// CHECK:STDOUT:   %t.patt.loc20_14.2 => constants.%t.patt.ee8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.5b3
// CHECK:STDOUT:   %.loc21_4 => constants.%.660
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.6ad
// CHECK:STDOUT:   %impl.elem0.loc21_4.2 => constants.%A.as.I.impl.Doit
// CHECK:STDOUT:   %specific_impl_fn.loc21_4.2 => constants.%A.as.I.impl.Doit
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%I.facet.9f8) {
// CHECK:STDOUT:   %T.patt.loc20_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc20_7.1 => constants.%I.facet.9f8
// CHECK:STDOUT:   %T.as_type.loc20_16.1 => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e39
// CHECK:STDOUT:   %t.param_patt.loc20_14.2 => constants.%t.param_patt.267
// CHECK:STDOUT:   %t.patt.loc20_14.2 => constants.%t.patt.062
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.2ce
// CHECK:STDOUT:   %.loc21_4 => constants.%.545
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.5b8
// CHECK:STDOUT:   %impl.elem0.loc21_4.2 => constants.%B.as.I.impl.Doit
// CHECK:STDOUT:   %specific_impl_fn.loc21_4.2 => constants.%B.as.I.impl.Doit
// CHECK:STDOUT: }
// CHECK:STDOUT:
