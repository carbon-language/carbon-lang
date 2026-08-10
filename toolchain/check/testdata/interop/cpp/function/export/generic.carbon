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
  fn Doit(self) -> Self;
}

class A {
  impl as I {
    fn Doit(unused self) -> Self {
      return {};
    }
  }
}
class B {
  impl as I {
    fn Doit(unused self) -> Self {
      return {};
    }
  }
}

fn F[T: I](t: T) -> T {
  return t.Doit();
}

inline Cpp '''
Carbon::B CallInterfaceF() {
  Carbon::A a;
  Carbon::B b;
  Carbon::F(a);
  return Carbon::F(b);
}
''';
//@dump-sem-ir-end

// --- fail_todo_non_type_generic.carbon
library "[[@TEST_NAME]]";
import Cpp;

// CHECK:STDERR: fail_todo_non_type_generic.carbon:[[@LINE+2]]:1-30: error: semantics TODO: `binding maps to a non-type template parameter` [SemanticsTodo]
// CHECK:STDERR:
fn NonTypeF[unused N: i32]() {}

inline Cpp '''
void CallNonTypeF() {
  // CHECK:STDERR: fail_todo_non_type_generic.carbon:[[@LINE+4]]:11-18: error: no member named 'NonTypeF' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_non_type_generic.carbon:[[@LINE+2]]:23: error: expected expression [CppInteropParseError]
  // CHECK:STDERR:
  Carbon::NonTypeF<3>();
}
''';

// --- generic_pointer.carbon
library "[[@TEST_NAME]]";
import Cpp;

fn PointerF[T: type](unused t: T*) {}

inline Cpp '''
void CallPointerF() {
  int x = 0;
  int* _Nonnull p = &x;
  Carbon::PointerF(p);
}
''';

// --- fail_todo_enclosing_generic.carbon
library "[[@TEST_NAME]]";
import Cpp;

class EnclosingA(T: type) {
  // CHECK:STDERR: fail_todo_enclosing_generic.carbon:[[@LINE+2]]:3-34: error: semantics TODO: `non-class non-namespace name scope` [SemanticsTodo]
  // CHECK:STDERR:
  fn MemberF[U: type](x: T, y: U);
}
alias EnclosingB = EnclosingA(i32);

inline Cpp '''
void CallEnclosingF() {
  // TODO: this currently fails because `EnclosingB` can't be imported, but the intent
  // is to test a generic function within an enclosing generic scope.
  //
  Carbon::EnclosingB b;
  // CHECK:STDERR: fail_todo_enclosing_generic.carbon:[[@LINE+2]]:5-11: error: no member named 'MemberF' in 'Carbon::EnclosingA<int>' [CppInteropParseError]
  // CHECK:STDERR:
  b.MemberF(1, 2);
}
''';

// --- fail_generic_unsupported_type.carbon
library "[[@TEST_NAME]]";
import Cpp;

// CHECK:STDERR: fail_generic_unsupported_type.carbon:[[@LINE+2]]:1-39: error: semantics TODO: `failed to import C++ type` [SemanticsTodo]
// CHECK:STDERR:
fn UnsupportedF[T: type](unused t: T) {}

inline Cpp '''
void CallUnsupportedF() {
  volatile int x;
  // CHECK:STDERR: fail_generic_unsupported_type.carbon:[[@LINE+3]]:3-22: error: no matching function for call to 'UnsupportedF' [CppInteropParseError]
  // CHECK:STDERR: fail_generic_unsupported_type.carbon:[[@LINE-6]]:39: note: candidate template ignored: substitution failure [with T = volatile int] [CppInteropParseNote]
  // CHECK:STDERR:
  Carbon::UnsupportedF(x);
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
// CHECK:STDOUT:   %self.patt.85e: %pattern_type.c60 = wrapper_binding_pattern self, %self.param_patt.a33 [symbolic]
// CHECK:STDOUT:   %.8fc: Core.Form = init_form %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %return.param_patt.434: %pattern_type.c60 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.44a: %pattern_type.c60 = return_slot_pattern %return.param_patt.434, %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.26f: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.11f: %I.WithSelf.Doit.type.26f = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.628: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Doit.decl [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %.b03: <witness> = impl_self_witness %A, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.ca2: <witness> = impl_witness @A.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7cd: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %.cbf: Core.Form = init_form %A [concrete]
// CHECK:STDOUT:   %return.param_patt.f0c: %pattern_type.9ef = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a6d: %pattern_type.9ef = return_slot_pattern %return.param_patt.f0c, %A [concrete]
// CHECK:STDOUT:   %A.as.I.impl.Doit.type: type = fn_type @A.as.I.impl.Doit [concrete]
// CHECK:STDOUT:   %A.as.I.impl.Doit: %A.as.I.impl.Doit.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.f31: %I.type = facet_value %A, (%I.impl_witness.ca2) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.808: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%I.facet.f31) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.ebf: %I.WithSelf.Doit.type.808 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %.d64: <witness> = impl_self_witness %B, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.b14: <witness> = impl_witness @B.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %self.param_patt.f1f: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4c3: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt.f1f [concrete]
// CHECK:STDOUT:   %.d83: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %return.param_patt.934: %pattern_type.e39 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.911: %pattern_type.e39 = return_slot_pattern %return.param_patt.934, %B [concrete]
// CHECK:STDOUT:   %B.as.I.impl.Doit.type: type = fn_type @B.as.I.impl.Doit [concrete]
// CHECK:STDOUT:   %B.as.I.impl.Doit: %B.as.I.impl.Doit.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.a68: %I.type = facet_value %B, (%I.impl_witness.b14) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.a07: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%I.facet.a68) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.9cf: %I.WithSelf.Doit.type.a07 = struct_value () [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt.5db: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.d48: %pattern_type.b3a = wrapper_binding_pattern t, %t.param_patt.5db [symbolic]
// CHECK:STDOUT:   %.0ce: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.b82: %pattern_type.b3a = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.2e6: %pattern_type.b3a = return_slot_pattern %return.param_patt.b82, %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.860: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.45d: %I.WithSelf.Doit.type.860 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.8a9: type = fn_type_with_self_type %I.WithSelf.Doit.type.860, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.8a9 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.58e: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b4c: %pattern_type.b3a = wrapper_binding_pattern self, %self.param_patt.58e [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Doit(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc16_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f70: %Destroy.type = facet_value %B, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.cda: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f70) [concrete]
// CHECK:STDOUT:   %.3d9: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.cda, %Destroy.facet.f70 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e5: %Destroy.type = facet_value %A, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.09e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6e5) [concrete]
// CHECK:STDOUT:   %.dca: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.09e, %Destroy.facet.6e5 [concrete]
// CHECK:STDOUT:   %t.param_patt.169: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.490: %pattern_type.9ef = wrapper_binding_pattern t, %t.param_patt.169 [concrete]
// CHECK:STDOUT:   %F.specific_fn.9d8: <specific function> = specific_function %F, @F(%I.facet.f31) [concrete]
// CHECK:STDOUT:   %t.param_patt.267: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.614: %pattern_type.e39 = wrapper_binding_pattern t, %t.param_patt.267 [concrete]
// CHECK:STDOUT:   %F.specific_fn.480: <specific function> = specific_function %F, @F(%I.facet.a68) [concrete]
// CHECK:STDOUT:   %.760: type = fn_type_with_self_type %I.WithSelf.Doit.type.808, %I.facet.f31 [concrete]
// CHECK:STDOUT:   %.10f: type = fn_type_with_self_type %I.WithSelf.Doit.type.a07, %I.facet.a68 [concrete]
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
// CHECK:STDOUT:     %T.patt.loc24_7.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc24_13.1: @F.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %t.param_patt.loc24_13.2 (constants.%t.param_patt.5db)]
// CHECK:STDOUT:     %t.patt.loc24_13.1: @F.%pattern_type (%pattern_type.b3a) = wrapper_binding_pattern t, %t.param_patt.loc24_13.1 [symbolic = %t.patt.loc24_13.2 (constants.%t.patt.d48)]
// CHECK:STDOUT:     %return.param_patt.loc24_21.1: @F.%pattern_type (%pattern_type.b3a) = out_param_pattern [symbolic = %return.param_patt.loc24_21.2 (constants.%return.param_patt.b82)]
// CHECK:STDOUT:     %return.patt.loc24_18.1: @F.%pattern_type (%pattern_type.b3a) = return_slot_pattern %return.param_patt.loc24_21.1, %.loc24_21.3 [symbolic = %return.patt.loc24_18.2 (constants.%return.patt.2e6)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc24_21: %I.type = name_ref T, %T.loc24_7.2 [symbolic = %T.loc24_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc24_21: type = facet_access_type %T.ref.loc24_21 [symbolic = %T.as_type.loc24_15.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc24_21.3: type = converted %T.ref.loc24_21, %T.as_type.loc24_21 [symbolic = %T.as_type.loc24_15.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc24_21.4: Core.Form = init_form %.loc24_21.3 [symbolic = %.loc24_21.2 (constants.%.0ce)]
// CHECK:STDOUT:     %.loc24_9: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24_7.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc24_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc24_15.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc24_15.1: type = splice_block %.loc24_15.2 [symbolic = %T.as_type.loc24_15.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc24_15: %I.type = name_ref T, %T.loc24_7.2 [symbolic = %T.loc24_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc24_15.2: type = facet_access_type %T.ref.loc24_15 [symbolic = %T.as_type.loc24_15.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc24_15.2: type = converted %T.ref.loc24_15, %T.as_type.loc24_15.2 [symbolic = %T.as_type.loc24_15.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc24_15.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%T.as_type.loc24_15.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.as_type.loc24_15.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %I.facet.loc24_23.1: %I.type = facet_value %A.decl, (constants.%I.impl_witness.ca2) [concrete = constants.%I.facet.f31]
// CHECK:STDOUT:   %.loc24_23.1: %I.type = converted %A.decl, %I.facet.loc24_23.1 [concrete = constants.%I.facet.f31]
// CHECK:STDOUT:   %I.facet.loc24_23.2: %I.type = facet_value %B.decl, (constants.%I.impl_witness.b14) [concrete = constants.%I.facet.a68]
// CHECK:STDOUT:   %.loc24_23.2: %I.type = converted %B.decl, %I.facet.loc24_23.2 [concrete = constants.%I.facet.a68]
// CHECK:STDOUT:   inline_cpp "Carbon::B CallInterfaceF() {\n  Carbon::A a;\n  Carbon::B b;\n  Carbon::F(a);\n  return Carbon::F(b);\n}\n"
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.Doit.decl: @I.WithSelf.%I.WithSelf.Doit.type (%I.WithSelf.Doit.type.26f) = fn_decl @I.WithSelf.Doit [symbolic = @I.WithSelf.%I.WithSelf.Doit (constants.%I.WithSelf.Doit.11f)] {
// CHECK:STDOUT:     %self.param_patt.loc6_11.1: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc6_11.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:     %self.patt.loc6_11.1: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = wrapper_binding_pattern self, %self.param_patt.loc6_11.1 [symbolic = %self.patt.loc6_11.2 (constants.%self.patt.85e)]
// CHECK:STDOUT:     %return.param_patt.loc6_20.1: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = out_param_pattern [symbolic = %return.param_patt.loc6_20.2 (constants.%return.param_patt.434)]
// CHECK:STDOUT:     %return.patt.loc6_17.1: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = return_slot_pattern %return.param_patt.loc6_20.1, %.loc6_20.2 [symbolic = %return.patt.loc6_17.2 (constants.%return.patt.44a)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc6_20: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:     %Self.as_type.loc6_20: type = facet_access_type %Self.ref.loc6_20 [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     %.loc6_20.2: type = converted %Self.ref.loc6_20, %Self.as_type.loc6_20 [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     %.loc6_20.3: Core.Form = init_form %.loc6_20.2 [symbolic = %.loc6_20.1 (constants.%.8fc)]
// CHECK:STDOUT:     %self.param: @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84) = value_param call_param0
// CHECK:STDOUT:     %.loc6_11.1: type = splice_block %.loc6_11.2 [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)] {
// CHECK:STDOUT:       %Self.ref.loc6_11: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:       %Self.as_type.loc6_11.2: type = facet_access_type %Self.ref.loc6_11 [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:       %.loc6_11.2: type = converted %Self.ref.loc6_11, %Self.as_type.loc6_11.2 [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84) = out_param call_param1
// CHECK:STDOUT:     %return: ref @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84) = return_slot %return.param
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
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7cd]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9ef = out_param_pattern [concrete = constants.%return.param_patt.f0c]
// CHECK:STDOUT:     %return.patt: %pattern_type.9ef = return_slot_pattern %return.param_patt, %Self.ref.loc11_29 [concrete = constants.%return.patt.a6d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc11_29: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %.loc11: Core.Form = init_form %Self.ref.loc11_29 [concrete = constants.%.cbf]
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc11_20: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %A = out_param call_param1
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%A.as.I.impl.Doit.decl), @A.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.ca2]
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
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4c3]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e39 = out_param_pattern [concrete = constants.%return.param_patt.934]
// CHECK:STDOUT:     %return.patt: %pattern_type.e39 = return_slot_pattern %return.param_patt, %Self.ref.loc18_29 [concrete = constants.%return.patt.911]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_29: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %Self.ref.loc18_29 [concrete = constants.%.d83]
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc18_20: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param1
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%B.as.I.impl.Doit.decl), @B.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.b14]
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
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @A.as.I.impl.%Self.ref, @I [concrete = constants.%.b03]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   impl_decl @B.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = impl_self_witness @B.as.I.impl.%Self.ref, @I [concrete = constants.%.d64]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.Doit(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:   %Self.as_type.loc6_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_11.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc6_11.1 [symbolic = %pattern_type (constants.%pattern_type.c60)]
// CHECK:STDOUT:   %self.param_patt.loc6_11.2: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc6_11.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:   %self.patt.loc6_11.2: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = wrapper_binding_pattern self, %self.param_patt.loc6_11.2 [symbolic = %self.patt.loc6_11.2 (constants.%self.patt.85e)]
// CHECK:STDOUT:   %.loc6_20.1: Core.Form = init_form %Self.as_type.loc6_11.1 [symbolic = %.loc6_20.1 (constants.%.8fc)]
// CHECK:STDOUT:   %return.param_patt.loc6_20.2: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = out_param_pattern [symbolic = %return.param_patt.loc6_20.2 (constants.%return.param_patt.434)]
// CHECK:STDOUT:   %return.patt.loc6_17.2: @I.WithSelf.Doit.%pattern_type (%pattern_type.c60) = return_slot_pattern %return.param_patt.loc6_20.2, %Self.as_type.loc6_11.1 [symbolic = %return.patt.loc6_17.2 (constants.%return.patt.44a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84)) -> out %return.param: @I.WithSelf.Doit.%Self.as_type.loc6_11.1 (%Self.as_type.c84);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.I.impl.Doit(%self.param: %A) -> out %return.param: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_15.2: init %A to %return.param = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc12_16: init %A = converted %.loc12_15.1, %.loc12_15.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   return %.loc12_16 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.I.impl.Doit(%self.param: %B) -> out %return.param: %B {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc19_15.2: init %B to %return.param = class_init () [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc19_16: init %B = converted %.loc19_15.1, %.loc19_15.2 [concrete = constants.%B.val]
// CHECK:STDOUT:   return %.loc19_16 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc24_7.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc24_7.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc24_7.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc24_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc24_15.1: type = facet_access_type %T.loc24_7.1 [symbolic = %T.as_type.loc24_15.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc24_15.1 [symbolic = %pattern_type (constants.%pattern_type.b3a)]
// CHECK:STDOUT:   %t.param_patt.loc24_13.2: @F.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %t.param_patt.loc24_13.2 (constants.%t.param_patt.5db)]
// CHECK:STDOUT:   %t.patt.loc24_13.2: @F.%pattern_type (%pattern_type.b3a) = wrapper_binding_pattern t, %t.param_patt.loc24_13.2 [symbolic = %t.patt.loc24_13.2 (constants.%t.patt.d48)]
// CHECK:STDOUT:   %.loc24_21.2: Core.Form = init_form %T.as_type.loc24_15.1 [symbolic = %.loc24_21.2 (constants.%.0ce)]
// CHECK:STDOUT:   %return.param_patt.loc24_21.2: @F.%pattern_type (%pattern_type.b3a) = out_param_pattern [symbolic = %return.param_patt.loc24_21.2 (constants.%return.param_patt.b82)]
// CHECK:STDOUT:   %return.patt.loc24_18.2: @F.%pattern_type (%pattern_type.b3a) = return_slot_pattern %return.param_patt.loc24_21.2, %T.as_type.loc24_15.1 [symbolic = %return.patt.loc24_18.2 (constants.%return.patt.2e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc24_15.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.WithSelf.Doit.type: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%T.loc24_7.1) [symbolic = %I.WithSelf.Doit.type (constants.%I.WithSelf.Doit.type.860)]
// CHECK:STDOUT:   %.loc25_11: type = fn_type_with_self_type %I.WithSelf.Doit.type, %T.loc24_7.1 [symbolic = %.loc25_11 (constants.%.8a9)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc24_7.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc25_11.2: @F.%.loc25_11 (%.8a9) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_11.2: <specific function> = specific_impl_function %impl.elem0.loc25_11.2, @I.WithSelf.Doit(%T.loc24_7.1) [symbolic = %specific_impl_fn.loc25_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc24_15.1 (%T.as_type)) -> out %return.param: @F.%T.as_type.loc24_15.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc24_15.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %Doit.ref: %I.assoc_type = name_ref Doit, @I.WithSelf.%assoc0 [concrete = constants.%assoc0.628]
// CHECK:STDOUT:     %impl.elem0.loc25_11.1: @F.%.loc25_11 (%.8a9) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc25_11: <bound method> = bound_method %t.ref, %impl.elem0.loc25_11.1
// CHECK:STDOUT:     %.loc25_17.1: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc24_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc25_17.2: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc24_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc25_11.1: <specific function> = specific_impl_function %impl.elem0.loc25_11.1, @I.WithSelf.Doit(constants.%T) [symbolic = %specific_impl_fn.loc25_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc25_17: <bound method> = bound_method %t.ref, %specific_impl_fn.loc25_11.1
// CHECK:STDOUT:     %.loc24_21.1: ref @F.%T.as_type.loc24_15.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %I.WithSelf.Doit.call: init @F.%T.as_type.loc24_15.1 (%T.as_type) to %.loc24_21.1 = call %bound_method.loc25_17(%t.ref)
// CHECK:STDOUT:     return %I.WithSelf.Doit.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.__destroy_thunk(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %impl.elem0: %.3d9 = impl_witness_access constants.%custom_witness.df9cc1.2, element0 [concrete = constants.%Destroy.Op.1a2547.2]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.param, %impl.elem0
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %bound_method(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_9.2(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.__destroy_thunk(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %impl.elem0: %.dca = impl_witness_access constants.%custom_witness.df9cc1.3, element0 [concrete = constants.%Destroy.Op.1a2547.3]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.param, %impl.elem0
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %bound_method(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F__carbon_thunk.26f2099a79512a5d(%_.param.loc24_23.1: ref %A, %_.param.loc24_23.2: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F__carbon_thunk.26f2099a79512a5d.ref: %F.type = name_ref F__carbon_thunk.26f2099a79512a5d, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %I.facet.loc24_23.1: %I.type = facet_value constants.%A, (constants.%I.impl_witness.ca2) [concrete = constants.%I.facet.f31]
// CHECK:STDOUT:   %.loc24_23.1: %I.type = converted constants.%A, %I.facet.loc24_23.1 [concrete = constants.%I.facet.f31]
// CHECK:STDOUT:   %I.facet.loc24_23.2: %I.type = facet_value constants.%A, (constants.%I.impl_witness.ca2) [concrete = constants.%I.facet.f31]
// CHECK:STDOUT:   %.loc24_23.2: %I.type = converted constants.%A, %I.facet.loc24_23.2 [concrete = constants.%I.facet.f31]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F__carbon_thunk.26f2099a79512a5d.ref, @F(constants.%I.facet.f31) [concrete = constants.%F.specific_fn.9d8]
// CHECK:STDOUT:   %.loc24_23.3: ref %A = splice_block %_.param.loc24_23.2 {}
// CHECK:STDOUT:   %.loc24_23.4: %A = acquire_value %_.param.loc24_23.1
// CHECK:STDOUT:   %F.call: init %A to %.loc24_23.3 = call %F.specific_fn(%.loc24_23.4)
// CHECK:STDOUT:   assign %_.param.loc24_23.2, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F__carbon_thunk.7a3b9f04ddf4269e(%_.param.loc24_23.1: ref %B, %_.param.loc24_23.2: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F__carbon_thunk.7a3b9f04ddf4269e.ref: %F.type = name_ref F__carbon_thunk.7a3b9f04ddf4269e, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %I.facet.loc24_23.1: %I.type = facet_value constants.%B, (constants.%I.impl_witness.b14) [concrete = constants.%I.facet.a68]
// CHECK:STDOUT:   %.loc24_23.1: %I.type = converted constants.%B, %I.facet.loc24_23.1 [concrete = constants.%I.facet.a68]
// CHECK:STDOUT:   %I.facet.loc24_23.2: %I.type = facet_value constants.%B, (constants.%I.impl_witness.b14) [concrete = constants.%I.facet.a68]
// CHECK:STDOUT:   %.loc24_23.2: %I.type = converted constants.%B, %I.facet.loc24_23.2 [concrete = constants.%I.facet.a68]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F__carbon_thunk.7a3b9f04ddf4269e.ref, @F(constants.%I.facet.a68) [concrete = constants.%F.specific_fn.480]
// CHECK:STDOUT:   %.loc24_23.3: ref %B = splice_block %_.param.loc24_23.2 {}
// CHECK:STDOUT:   %.loc24_23.4: %B = acquire_value %_.param.loc24_23.1
// CHECK:STDOUT:   %F.call: init %B to %.loc24_23.3 = call %F.specific_fn(%.loc24_23.4)
// CHECK:STDOUT:   assign %_.param.loc24_23.2, %F.call
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
// CHECK:STDOUT:   %self.patt.loc6_11.2 => constants.%self.patt.85e
// CHECK:STDOUT:   %.loc6_20.1 => constants.%.8fc
// CHECK:STDOUT:   %return.param_patt.loc6_20.2 => constants.%return.param_patt.434
// CHECK:STDOUT:   %return.patt.loc6_17.2 => constants.%return.patt.44a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.f31) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.f31
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.808
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.ebf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%I.facet.f31) {
// CHECK:STDOUT:   %Self => constants.%I.facet.f31
// CHECK:STDOUT:   %Self.as_type.loc6_11.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ef
// CHECK:STDOUT:   %self.param_patt.loc6_11.2 => constants.%self.param_patt.04e
// CHECK:STDOUT:   %self.patt.loc6_11.2 => constants.%self.patt.7cd
// CHECK:STDOUT:   %.loc6_20.1 => constants.%.cbf
// CHECK:STDOUT:   %return.param_patt.loc6_20.2 => constants.%return.param_patt.f0c
// CHECK:STDOUT:   %return.patt.loc6_17.2 => constants.%return.patt.a6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.a68) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.a68
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.a07
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.9cf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.Doit(constants.%I.facet.a68) {
// CHECK:STDOUT:   %Self => constants.%I.facet.a68
// CHECK:STDOUT:   %Self.as_type.loc6_11.1 => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e39
// CHECK:STDOUT:   %self.param_patt.loc6_11.2 => constants.%self.param_patt.f1f
// CHECK:STDOUT:   %self.patt.loc6_11.2 => constants.%self.patt.4c3
// CHECK:STDOUT:   %.loc6_20.1 => constants.%.d83
// CHECK:STDOUT:   %return.param_patt.loc6_20.2 => constants.%return.param_patt.934
// CHECK:STDOUT:   %return.patt.loc6_17.2 => constants.%return.patt.911
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc24_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc24_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc24_15.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %t.param_patt.loc24_13.2 => constants.%t.param_patt.5db
// CHECK:STDOUT:   %t.patt.loc24_13.2 => constants.%t.patt.d48
// CHECK:STDOUT:   %.loc24_21.2 => constants.%.0ce
// CHECK:STDOUT:   %return.param_patt.loc24_21.2 => constants.%return.param_patt.b82
// CHECK:STDOUT:   %return.patt.loc24_18.2 => constants.%return.patt.2e6
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
// CHECK:STDOUT:   %self.patt.loc6_11.2 => constants.%self.patt.b4c
// CHECK:STDOUT:   %.loc6_20.1 => constants.%.0ce
// CHECK:STDOUT:   %return.param_patt.loc6_20.2 => constants.%return.param_patt.b82
// CHECK:STDOUT:   %return.patt.loc6_17.2 => constants.%return.patt.2e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(file.%.loc24_23.1) {
// CHECK:STDOUT:   %T.patt.loc24_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc24_7.1 => constants.%I.facet.f31
// CHECK:STDOUT:   %T.as_type.loc24_15.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ef
// CHECK:STDOUT:   %t.param_patt.loc24_13.2 => constants.%t.param_patt.169
// CHECK:STDOUT:   %t.patt.loc24_13.2 => constants.%t.patt.490
// CHECK:STDOUT:   %.loc24_21.2 => constants.%.cbf
// CHECK:STDOUT:   %return.param_patt.loc24_21.2 => constants.%return.param_patt.f0c
// CHECK:STDOUT:   %return.patt.loc24_18.2 => constants.%return.patt.a6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%I.facet.f31) {
// CHECK:STDOUT:   %T.patt.loc24_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc24_7.1 => constants.%I.facet.f31
// CHECK:STDOUT:   %T.as_type.loc24_15.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ef
// CHECK:STDOUT:   %t.param_patt.loc24_13.2 => constants.%t.param_patt.169
// CHECK:STDOUT:   %t.patt.loc24_13.2 => constants.%t.patt.490
// CHECK:STDOUT:   %.loc24_21.2 => constants.%.cbf
// CHECK:STDOUT:   %return.param_patt.loc24_21.2 => constants.%return.param_patt.f0c
// CHECK:STDOUT:   %return.patt.loc24_18.2 => constants.%return.patt.a6d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.808
// CHECK:STDOUT:   %.loc25_11 => constants.%.760
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.ca2
// CHECK:STDOUT:   %impl.elem0.loc25_11.2 => constants.%A.as.I.impl.Doit
// CHECK:STDOUT:   %specific_impl_fn.loc25_11.2 => constants.%A.as.I.impl.Doit
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(file.%.loc24_23.2) {
// CHECK:STDOUT:   %T.patt.loc24_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc24_7.1 => constants.%I.facet.a68
// CHECK:STDOUT:   %T.as_type.loc24_15.1 => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e39
// CHECK:STDOUT:   %t.param_patt.loc24_13.2 => constants.%t.param_patt.267
// CHECK:STDOUT:   %t.patt.loc24_13.2 => constants.%t.patt.614
// CHECK:STDOUT:   %.loc24_21.2 => constants.%.d83
// CHECK:STDOUT:   %return.param_patt.loc24_21.2 => constants.%return.param_patt.934
// CHECK:STDOUT:   %return.patt.loc24_18.2 => constants.%return.patt.911
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%I.facet.a68) {
// CHECK:STDOUT:   %T.patt.loc24_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc24_7.1 => constants.%I.facet.a68
// CHECK:STDOUT:   %T.as_type.loc24_15.1 => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e39
// CHECK:STDOUT:   %t.param_patt.loc24_13.2 => constants.%t.param_patt.267
// CHECK:STDOUT:   %t.patt.loc24_13.2 => constants.%t.patt.614
// CHECK:STDOUT:   %.loc24_21.2 => constants.%.d83
// CHECK:STDOUT:   %return.param_patt.loc24_21.2 => constants.%return.param_patt.934
// CHECK:STDOUT:   %return.patt.loc24_18.2 => constants.%return.patt.911
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.a07
// CHECK:STDOUT:   %.loc25_11 => constants.%.10f
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.b14
// CHECK:STDOUT:   %impl.elem0.loc25_11.2 => constants.%B.as.I.impl.Doit
// CHECK:STDOUT:   %specific_impl_fn.loc25_11.2 => constants.%B.as.I.impl.Doit
// CHECK:STDOUT: }
// CHECK:STDOUT:
