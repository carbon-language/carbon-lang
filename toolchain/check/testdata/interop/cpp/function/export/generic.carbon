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

// --- generic_facet.carbon
library "[[@TEST_NAME]]";
import Cpp;

//@dump-sem-ir-begin
class D {}
impl D as Core.Copy {
  fn Op(unused self: D) -> D { return {} as D; }
}
fn MakeD() -> D { return {} as D; }

class C(T: Core.Copy & Core.Destroy) {
  var t: T;

  impl as Core.Copy {
    fn Op(self: Self) -> Self {
      let c: Self = {.t = self.t};
      return c;
    }
  }
}
fn MakeC[T: Core.Copy & Core.Destroy](t: T) -> C(T) {
  let c: C(T) = {.t = t};
  return c;
}

inline Cpp '''
void CallMake() {
  Carbon::MakeC(Carbon::MakeD());
}
''';
//@dump-sem-ir-end

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
  // CHECK:STDERR: fail_todo_enclosing_generic.carbon:[[@LINE+2]]:5: error: no member named 'MemberF' in 'Carbon::Main::EnclosingA<int>' [CppInteropParseError]
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

// CHECK:STDOUT: --- generic_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %.e69: <witness> = impl_self_witness %D, @Copy [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.548: <witness> = impl_witness @D.as.Copy.impl.%Copy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %self.param_patt.f16: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.752: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt.f16 [concrete]
// CHECK:STDOUT:   %.592: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %return.param_patt.eae: %pattern_type.d8d = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e38: %pattern_type.d8d = return_slot_pattern %return.param_patt.eae, %D [concrete]
// CHECK:STDOUT:   %D.as.Copy.impl.Op.type: type = fn_type @D.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D.as.Copy.impl.Op: %D.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.48d: %Copy.type = facet_value %D, (%Copy.impl_witness.548) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5bb: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.48d) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %MakeD.type: type = fn_type @MakeD [concrete]
// CHECK:STDOUT:   %MakeD: %MakeD.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.7f7: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.7f7 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.1a6: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.b4e: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.1a6, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Copy.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.143: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.patt.256: %pattern_type.143 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.dc4: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.210: type = class_type @C, @C(%T.dc4) [symbolic]
// CHECK:STDOUT:   %T.as_type.5d6: type = facet_access_type %T.dc4 [symbolic]
// CHECK:STDOUT:   %require_complete.091: <witness> = require_complete_type %T.as_type.5d6 [symbolic]
// CHECK:STDOUT:   %pattern_type.0e9: type = pattern_type %T.as_type.5d6 [symbolic]
// CHECK:STDOUT:   %C.elem.731: type = unbound_element_type %C.210, %T.as_type.5d6 [symbolic]
// CHECK:STDOUT:   %.8ff: <witness> = impl_self_witness %C.210, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8ad: <witness> = impl_witness @C.as.Copy.impl.%Copy.impl_witness_table, @C.as.Copy.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %pattern_type.0b5: type = pattern_type %C.210 [symbolic]
// CHECK:STDOUT:   %self.param_patt.27c: %pattern_type.0b5 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.fb3: %pattern_type.0b5 = wrapper_binding_pattern self, %self.param_patt.27c [symbolic]
// CHECK:STDOUT:   %.501: Core.Form = init_form %C.210 [symbolic]
// CHECK:STDOUT:   %return.param_patt.2c6: %pattern_type.0b5 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.a82: %pattern_type.0b5 = return_slot_pattern %return.param_patt.2c6, %C.210 [symbolic]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type.c14: type = fn_type @C.as.Copy.impl.Op, @C.as.Copy.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.4d5: %C.as.Copy.impl.Op.type.c14 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.t.87b: type = struct_type {.t: %T.as_type.5d6} [symbolic]
// CHECK:STDOUT:   %complete_type.856: <witness> = complete_type_witness %struct_type.t.87b [symbolic]
// CHECK:STDOUT:   %require_complete.bab: <witness> = require_complete_type %C.210 [symbolic]
// CHECK:STDOUT:   %c.patt.6d2: %pattern_type.0b5 = value_binding_pattern c [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.fd8: <witness> = lookup_impl_witness %T.dc4, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.40c: %Copy.type = facet_value %T.as_type.5d6, (%Copy.lookup_impl_witness.fd8) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f43: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.40c) [symbolic]
// CHECK:STDOUT:   %.81a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f43, %Copy.facet.40c [symbolic]
// CHECK:STDOUT:   %impl.elem0.35e: %.81a = impl_witness_access %Copy.lookup_impl_witness.fd8, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f62: <specific function> = specific_impl_function %impl.elem0.35e, @Copy.WithSelf.Op(%Copy.facet.40c) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.006: <witness> = lookup_impl_witness %C.210, @Copy [symbolic]
// CHECK:STDOUT:   %.e8c: require_specific_def_type = require_specific_def @C.as.Copy.impl(%T.dc4) [symbolic]
// CHECK:STDOUT:   %Copy.facet.c47: %Copy.type = facet_value %C.210, (%Copy.lookup_impl_witness.006) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.82d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.c47) [symbolic]
// CHECK:STDOUT:   %.df2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.82d, %Copy.facet.c47 [symbolic]
// CHECK:STDOUT:   %impl.elem0.83f: %.df2 = impl_witness_access %Copy.lookup_impl_witness.006, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f64: <specific function> = specific_impl_function %impl.elem0.83f, @Copy.WithSelf.Op(%Copy.facet.c47) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f09: <witness> = lookup_impl_witness %C.210, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dc5: %Destroy.type = facet_value %C.210, (%Destroy.lookup_impl_witness.f09) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.7ac: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.dc5) [symbolic]
// CHECK:STDOUT:   %.591: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.7ac, %Destroy.facet.dc5 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.591 = impl_witness_access %Destroy.lookup_impl_witness.f09, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.03e: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.dc5) [symbolic]
// CHECK:STDOUT:   %t.param_patt.081: %pattern_type.0e9 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.4d1: %pattern_type.0e9 = wrapper_binding_pattern t, %t.param_patt.081 [symbolic]
// CHECK:STDOUT:   %MakeC.type: type = fn_type @MakeC [concrete]
// CHECK:STDOUT:   %MakeC: %MakeC.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b8e: %pattern_type.d8d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.86b: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt.b8e [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc5_9.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc5_9.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc5_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc5_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc5_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %Destroy.WithSelf.SubobjectDestroy.d01daf.2, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %D, (%Copy.impl_witness.548, %custom_witness.f8f19d.2) [concrete]
// CHECK:STDOUT:   %t.param_patt.862: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.0d1: %pattern_type.d8d = wrapper_binding_pattern t, %t.param_patt.862 [concrete]
// CHECK:STDOUT:   %C.dac: type = class_type @C, @C(%facet_value) [concrete]
// CHECK:STDOUT:   %.01e: Core.Form = init_form %C.dac [concrete]
// CHECK:STDOUT:   %pattern_type.1ce: type = pattern_type %C.dac [concrete]
// CHECK:STDOUT:   %return.param_patt.f26: %pattern_type.1ce = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.52d: %pattern_type.1ce = return_slot_pattern %return.param_patt.f26, %C.dac [concrete]
// CHECK:STDOUT:   %C.elem.54d: type = unbound_element_type %C.dac, %D [concrete]
// CHECK:STDOUT:   %struct_type.t.9ba: type = struct_type {.t: %D} [concrete]
// CHECK:STDOUT:   %complete_type.b13: <witness> = complete_type_witness %struct_type.t.9ba [concrete]
// CHECK:STDOUT:   %MakeC.specific_fn: <specific function> = specific_function %MakeC, @MakeC(%facet_value) [concrete]
// CHECK:STDOUT:   %c.patt.50c: %pattern_type.1ce = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %.e9c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5bb, %Copy.facet.48d [concrete]
// CHECK:STDOUT:   %.ede: <witness> = impl_self_witness %C.dac, @Copy [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a7d: <witness> = impl_witness @C.as.Copy.impl.%Copy.impl_witness_table, @C.as.Copy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type.de5: type = fn_type @C.as.Copy.impl.Op, @C.as.Copy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.2ba: %C.as.Copy.impl.Op.type.de5 = struct_value () [concrete]
// CHECK:STDOUT:   %.50c: require_specific_def_type = require_specific_def @C.as.Copy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %Copy.facet.569: %Copy.type = facet_value %C.dac, (%Copy.impl_witness.a7d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6a6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.569) [concrete]
// CHECK:STDOUT:   %.0a2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6a6, %Copy.facet.569 [concrete]
// CHECK:STDOUT:   %self.param_patt.9bb: %pattern_type.1ce = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.89a: %pattern_type.1ce = wrapper_binding_pattern self, %self.param_patt.9bb [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %C.as.Copy.impl.Op.2ba, @C.as.Copy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %pattern_type.2c3: type = pattern_type %struct_type.t.9ba [concrete]
// CHECK:STDOUT:   %self.param_patt.17e: %pattern_type.2c3 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.dca: %pattern_type.2c3 = wrapper_binding_pattern self, %self.param_patt.17e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_24.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22_24.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.580: %pattern_type.1ce = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.591: %pattern_type.1ce = wrapper_binding_pattern self, %self.param_patt.580 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_24.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc22_24.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22_24.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.4: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.4, %Destroy.WithSelf.SubobjectDestroy.d01daf.4, %Destroy.WithSelf.SelfDestruct.db3fdb.4), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d0b: %Destroy.type = facet_value %C.dac, (%custom_witness.f8f19d.4) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.70e: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.d0b) [concrete]
// CHECK:STDOUT:   %.c6c: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.70e, %Destroy.facet.d0b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc5_9.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc5_9.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc5_9.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = ref_param_pattern [concrete = constants.%self.param_patt.b8e]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.86b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %D = ref_param call_param0
// CHECK:STDOUT:     %self: ref %D = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc5_9.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = ref_param_pattern [concrete = constants.%self.param_patt.b8e]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.86b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %D = ref_param call_param0
// CHECK:STDOUT:     %self: ref %D = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_24.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2c3 = ref_param_pattern [concrete = constants.%self.param_patt.17e]
// CHECK:STDOUT:     %self.patt: %pattern_type.2c3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.dca]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.t.9ba = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.t.9ba = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc22_24.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2c3 = ref_param_pattern [concrete = constants.%self.param_patt.17e]
// CHECK:STDOUT:     %self.patt: %pattern_type.2c3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.dca]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.t.9ba = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.t.9ba = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_24.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ce = ref_param_pattern [concrete = constants.%self.param_patt.580]
// CHECK:STDOUT:     %self.patt: %pattern_type.1ce = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.591]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.dac = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.dac = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc22_24.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ce = ref_param_pattern [concrete = constants.%self.param_patt.580]
// CHECK:STDOUT:     %self.patt: %pattern_type.1ce = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.591]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.dac = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.dac = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @D.as.Copy.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @D.as.Copy.impl.%D.ref, @Copy [concrete = constants.%.e69]
// CHECK:STDOUT:   %MakeD.decl: %MakeD.type = fn_decl @MakeD [concrete = constants.%MakeD] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d8d = out_param_pattern [concrete = constants.%return.param_patt.eae]
// CHECK:STDOUT:     %return.patt: %pattern_type.d8d = return_slot_pattern %return.param_patt, %D.ref.loc9_15 [concrete = constants.%return.patt.e38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_15: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_15.2: Core.Form = init_form %D.ref.loc9_15 [concrete = constants.%.592]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc11_10.1: %pattern_type.143 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt.256)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_22.1: type = splice_block %.loc11_22.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref.loc11_12: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:       %Core.ref.loc11_24: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0: %.b4e = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %Copy.ref, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%Copy.ref, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc11_22.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc11_22.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc11_22.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_10.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc11_10.1 (constants.%T.dc4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeC.decl: %MakeC.type = fn_decl @MakeC [concrete = constants.%MakeC] {
// CHECK:STDOUT:     %T.patt.loc21_11.1: %pattern_type.143 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_11.2 (constants.%T.patt.256)]
// CHECK:STDOUT:     %t.param_patt.loc21_40.1: @MakeC.%pattern_type.loc21_40 (%pattern_type.0e9) = value_param_pattern [symbolic = %t.param_patt.loc21_40.2 (constants.%t.param_patt.081)]
// CHECK:STDOUT:     %t.patt.loc21_40.1: @MakeC.%pattern_type.loc21_40 (%pattern_type.0e9) = wrapper_binding_pattern t, %t.param_patt.loc21_40.1 [symbolic = %t.patt.loc21_40.2 (constants.%t.patt.4d1)]
// CHECK:STDOUT:     %return.param_patt.loc21_51.1: @MakeC.%pattern_type.loc21_51 (%pattern_type.0b5) = out_param_pattern [symbolic = %return.param_patt.loc21_51.2 (constants.%return.param_patt.2c6)]
// CHECK:STDOUT:     %return.patt.loc21_45.1: @MakeC.%pattern_type.loc21_51 (%pattern_type.0b5) = return_slot_pattern %return.param_patt.loc21_51.1, %C.loc21_51.2 [symbolic = %return.patt.loc21_45.2 (constants.%return.patt.a82)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc21: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc21_50: %facet_type = name_ref T, %T.loc21_11.2 [symbolic = %T.loc21_11.1 (constants.%T.dc4)]
// CHECK:STDOUT:     %C.loc21_51.2: type = class_type @C, @C(constants.%T.dc4) [symbolic = %C.loc21_51.1 (constants.%C.210)]
// CHECK:STDOUT:     %.loc21_51.3: Core.Form = init_form %C.loc21_51.2 [symbolic = %.loc21_51.2 (constants.%.501)]
// CHECK:STDOUT:     %.loc21_23.1: type = splice_block %.loc21_23.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref.loc21_13: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:       %Core.ref.loc21_25: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc21: %.b4e = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc21: <bound method> = bound_method %Copy.ref, %impl.elem0.loc21 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc21(%Copy.ref, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc21_23.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc21_23.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc21_23.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_11.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc21_11.1 (constants.%T.dc4)]
// CHECK:STDOUT:     %t.param: @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6) = value_param call_param0
// CHECK:STDOUT:     %.loc21_42.1: type = splice_block %.loc21_42.2 [symbolic = %T.as_type.loc21_42.1 (constants.%T.as_type.5d6)] {
// CHECK:STDOUT:       %T.ref.loc21_42: %facet_type = name_ref T, %T.loc21_11.2 [symbolic = %T.loc21_11.1 (constants.%T.dc4)]
// CHECK:STDOUT:       %T.as_type.loc21_42.2: type = facet_access_type %T.ref.loc21_42 [symbolic = %T.as_type.loc21_42.1 (constants.%T.as_type.5d6)]
// CHECK:STDOUT:       %.loc21_42.2: type = converted %T.ref.loc21_42, %T.as_type.loc21_42.2 [symbolic = %T.as_type.loc21_42.1 (constants.%T.as_type.5d6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6) = wrapper_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @MakeC.%C.loc21_51.1 (%C.210) = out_param call_param1
// CHECK:STDOUT:     %return: ref @MakeC.%C.loc21_51.1 (%C.210) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %D.decl, (constants.%Copy.impl_witness.548, constants.%custom_witness.f8f19d.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc21: %facet_type = converted %D.decl, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   inline_cpp "void CallMake() {\n  Carbon::MakeC(Carbon::MakeD());\n}\n"
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Copy.impl: %D.ref as %Copy.ref {
// CHECK:STDOUT:   %D.as.Copy.impl.Op.decl: %D.as.Copy.impl.Op.type = fn_decl @D.as.Copy.impl.Op [concrete = constants.%D.as.Copy.impl.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = value_param_pattern [concrete = constants.%self.param_patt.f16]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.752]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d8d = out_param_pattern [concrete = constants.%return.param_patt.eae]
// CHECK:STDOUT:     %return.patt: %pattern_type.d8d = return_slot_pattern %return.param_patt, %D.ref.loc7_28 [concrete = constants.%return.patt.e38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc7_28: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc7_28.2: Core.Form = init_form %D.ref.loc7_28 [concrete = constants.%.592]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref.loc7_22: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.impl_witness_table = impl_witness_table (%D.as.Copy.impl.Op.decl), @D.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Copy.impl_witness: <witness> = impl_witness %Copy.impl_witness_table [concrete = constants.%Copy.impl_witness.548]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Op = %D.as.Copy.impl.Op.decl
// CHECK:STDOUT:   extend %Copy.ref
// CHECK:STDOUT:   witness = %Copy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Copy.impl(@C.%T.loc11_10.2: %facet_type) {
// CHECK:STDOUT:   %T: %facet_type = symbolic_binding T, 0 [symbolic = %T (constants.%T.dc4)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness %C, @Copy [symbolic = %.loc14 (constants.%.8ff)]
// CHECK:STDOUT:   %Copy.impl_witness.loc14_21.2: <witness> = impl_witness %Copy.impl_witness_table, @C.as.Copy.impl(%T) [symbolic = %Copy.impl_witness.loc14_21.2 (constants.%Copy.impl_witness.8ad)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type: type = fn_type @C.as.Copy.impl.Op, @C.as.Copy.impl(%T) [symbolic = %C.as.Copy.impl.Op.type (constants.%C.as.Copy.impl.Op.type.c14)]
// CHECK:STDOUT:   %C.as.Copy.impl.Op: @C.as.Copy.impl.%C.as.Copy.impl.Op.type (%C.as.Copy.impl.Op.type.c14) = struct_value () [symbolic = %C.as.Copy.impl.Op (constants.%C.as.Copy.impl.Op.4d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Copy.ref {
// CHECK:STDOUT:     %C.as.Copy.impl.Op.decl: @C.as.Copy.impl.%C.as.Copy.impl.Op.type (%C.as.Copy.impl.Op.type.c14) = fn_decl @C.as.Copy.impl.Op [symbolic = @C.as.Copy.impl.%C.as.Copy.impl.Op (constants.%C.as.Copy.impl.Op.4d5)] {
// CHECK:STDOUT:       %self.param_patt.loc15_15.1: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = value_param_pattern [symbolic = %self.param_patt.loc15_15.2 (constants.%self.param_patt.27c)]
// CHECK:STDOUT:       %self.patt.loc15_15.1: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = wrapper_binding_pattern self, %self.param_patt.loc15_15.1 [symbolic = %self.patt.loc15_15.2 (constants.%self.patt.fb3)]
// CHECK:STDOUT:       %return.param_patt.loc15_26.1: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = out_param_pattern [symbolic = %return.param_patt.loc15_26.2 (constants.%return.param_patt.2c6)]
// CHECK:STDOUT:       %return.patt.loc15_23.1: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = return_slot_pattern %return.param_patt.loc15_26.1, %Self.ref.loc15_26 [symbolic = %return.patt.loc15_23.2 (constants.%return.patt.a82)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc15_26.3: type = specific_constant constants.%C.210, @C(constants.%T.dc4) [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:       %Self.ref.loc15_26: type = name_ref Self, %.loc15_26.3 [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:       %.loc15_26.4: Core.Form = init_form %Self.ref.loc15_26 [symbolic = %.loc15_26.2 (constants.%.501)]
// CHECK:STDOUT:       %self.param: @C.as.Copy.impl.Op.%C (%C.210) = value_param call_param0
// CHECK:STDOUT:       %.loc15_17.1: type = splice_block %Self.ref.loc15_17 [symbolic = %C (constants.%C.210)] {
// CHECK:STDOUT:         %.loc15_17.2: type = specific_constant constants.%C.210, @C(constants.%T.dc4) [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:         %Self.ref.loc15_17: type = name_ref Self, %.loc15_17.2 [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Copy.impl.Op.%C (%C.210) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @C.as.Copy.impl.Op.%C (%C.210) = out_param call_param1
// CHECK:STDOUT:       %return: ref @C.as.Copy.impl.Op.%C (%C.210) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Copy.impl_witness_table = impl_witness_table (%C.as.Copy.impl.Op.decl), @C.as.Copy.impl [concrete]
// CHECK:STDOUT:     %Copy.impl_witness.loc14_21.1: <witness> = impl_witness %Copy.impl_witness_table, @C.as.Copy.impl(constants.%T.dc4) [symbolic = %Copy.impl_witness.loc14_21.2 (constants.%Copy.impl_witness.8ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Copy.impl.Op.decl
// CHECK:STDOUT:     extend %Copy.ref
// CHECK:STDOUT:     witness = %Copy.impl_witness.loc14_21.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc11_10.2: %facet_type) {
// CHECK:STDOUT:   %T.patt.loc11_10.2: %pattern_type.143 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt.256)]
// CHECK:STDOUT:   %T.loc11_10.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc11_10.1 (constants.%T.dc4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_10.2: type = facet_access_type %T.loc11_10.1 [symbolic = %T.as_type.loc12_10.2 (constants.%T.as_type.5d6)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc12_10.2 [symbolic = %require_complete (constants.%require_complete.091)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc11_10.1) [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T.as_type.loc12_10.2 [symbolic = %C.elem (constants.%C.elem.731)]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: @C.%T.as_type.loc12_10.2 (%T.as_type.5d6)} [symbolic = %struct_type.t (constants.%struct_type.t.87b)]
// CHECK:STDOUT:   %complete_type.loc20_1.2: <witness> = complete_type_witness %struct_type.t [symbolic = %complete_type.loc20_1.2 (constants.%complete_type.856)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %field_decl: @C.%C.elem (%C.elem.731) = field_decl t, element0, %.loc12 in [concrete] {
// CHECK:STDOUT:       %T.ref: %facet_type = name_ref T, %T.loc11_10.2 [symbolic = %T.loc11_10.1 (constants.%T.dc4)]
// CHECK:STDOUT:       %T.as_type.loc12_10.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_10.2 (constants.%T.as_type.5d6)]
// CHECK:STDOUT:       %.loc12: type = converted %T.ref, %T.as_type.loc12_10.1 [symbolic = %T.as_type.loc12_10.2 (constants.%T.as_type.5d6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     impl_decl @C.as.Copy.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.210 [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc14: <witness> = impl_self_witness @C.as.Copy.impl.%Self.ref, @Copy [symbolic = @C.as.Copy.impl.%.loc14 (constants.%.8ff)]
// CHECK:STDOUT:     %complete_type.loc20_1.1: <witness> = complete_type_witness constants.%struct_type.t.87b [symbolic = %complete_type.loc20_1.2 (constants.%complete_type.856)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc20_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.210
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .t = %field_decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Copy.impl.Op(%self.param: %D) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_40.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %D.ref.loc7_45: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %.loc7_28.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc7_40.2: init %D to %.loc7_28.1 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc7_42: init %D = converted %.loc7_40.1, %.loc7_40.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc7_42 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeD() -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %D.ref.loc9_32: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %.loc9_15.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc9_27.2: init %D to %.loc9_15.1 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc9_29: init %D = converted %.loc9_27.1, %.loc9_27.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc9_29 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Copy.impl.Op(@C.%T.loc11_10.2: %facet_type) {
// CHECK:STDOUT:   %T: %facet_type = symbolic_binding T, 0 [symbolic = %T (constants.%T.dc4)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [symbolic = %pattern_type (constants.%pattern_type.0b5)]
// CHECK:STDOUT:   %self.param_patt.loc15_15.2: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = value_param_pattern [symbolic = %self.param_patt.loc15_15.2 (constants.%self.param_patt.27c)]
// CHECK:STDOUT:   %self.patt.loc15_15.2: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = wrapper_binding_pattern self, %self.param_patt.loc15_15.2 [symbolic = %self.patt.loc15_15.2 (constants.%self.patt.fb3)]
// CHECK:STDOUT:   %.loc15_26.2: Core.Form = init_form %C [symbolic = %.loc15_26.2 (constants.%.501)]
// CHECK:STDOUT:   %return.param_patt.loc15_26.2: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = out_param_pattern [symbolic = %return.param_patt.loc15_26.2 (constants.%return.param_patt.2c6)]
// CHECK:STDOUT:   %return.patt.loc15_23.2: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = return_slot_pattern %return.param_patt.loc15_26.2, %C [symbolic = %return.patt.loc15_23.2 (constants.%return.patt.a82)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %C [symbolic = %require_complete.loc15 (constants.%require_complete.bab)]
// CHECK:STDOUT:   %c.patt.loc16_12.2: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = value_binding_pattern c [symbolic = %c.patt.loc16_12.2 (constants.%c.patt.6d2)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic = %T.as_type (constants.%T.as_type.5d6)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %T.as_type [symbolic = %C.elem (constants.%C.elem.731)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %T.as_type [symbolic = %require_complete.loc16 (constants.%require_complete.091)]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: @C.as.Copy.impl.Op.%T.as_type (%T.as_type.5d6)} [symbolic = %struct_type.t (constants.%struct_type.t.87b)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc16: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness.loc16 (constants.%Copy.lookup_impl_witness.fd8)]
// CHECK:STDOUT:   %Copy.facet.loc16_31.3: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.loc16) [symbolic = %Copy.facet.loc16_31.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc16: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc16_31.3) [symbolic = %Copy.WithSelf.Op.type.loc16 (constants.%Copy.WithSelf.Op.type.f43)]
// CHECK:STDOUT:   %.loc16_31.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.loc16, %Copy.facet.loc16_31.3 [symbolic = %.loc16_31.5 (constants.%.81a)]
// CHECK:STDOUT:   %impl.elem0.loc16_31.2: @C.as.Copy.impl.Op.%.loc16_31.5 (%.81a) = impl_witness_access %Copy.lookup_impl_witness.loc16, element0 [symbolic = %impl.elem0.loc16_31.2 (constants.%impl.elem0.35e)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_31.2: <specific function> = specific_impl_function %impl.elem0.loc16_31.2, @Copy.WithSelf.Op(%Copy.facet.loc16_31.3) [symbolic = %specific_impl_fn.loc16_31.2 (constants.%specific_impl_fn.f62)]
// CHECK:STDOUT:   %.loc17_14.3: require_specific_def_type = require_specific_def @C.as.Copy.impl(%T) [symbolic = %.loc17_14.3 (constants.%.e8c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc17: <witness> = lookup_impl_witness %C, @Copy [symbolic = %Copy.lookup_impl_witness.loc17 (constants.%Copy.lookup_impl_witness.006)]
// CHECK:STDOUT:   %Copy.facet.loc17_14.3: %Copy.type = facet_value %C, (%Copy.lookup_impl_witness.loc17) [symbolic = %Copy.facet.loc17_14.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc17: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc17_14.3) [symbolic = %Copy.WithSelf.Op.type.loc17 (constants.%Copy.WithSelf.Op.type.82d)]
// CHECK:STDOUT:   %.loc17_14.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.loc17, %Copy.facet.loc17_14.3 [symbolic = %.loc17_14.4 (constants.%.df2)]
// CHECK:STDOUT:   %impl.elem0.loc17_14.2: @C.as.Copy.impl.Op.%.loc17_14.4 (%.df2) = impl_witness_access %Copy.lookup_impl_witness.loc17, element0 [symbolic = %impl.elem0.loc17_14.2 (constants.%impl.elem0.83f)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_14.2: <specific function> = specific_impl_function %impl.elem0.loc17_14.2, @Copy.WithSelf.Op(%Copy.facet.loc17_14.3) [symbolic = %specific_impl_fn.loc17_14.2 (constants.%specific_impl_fn.f64)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.f09)]
// CHECK:STDOUT:   %Destroy.facet.loc16_33.3: %Destroy.type = facet_value %C, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc16_33.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.loc16_33.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.7ac)]
// CHECK:STDOUT:   %.loc16_33.12: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.loc16_33.3 [symbolic = %.loc16_33.12 (constants.%.591)]
// CHECK:STDOUT:   %impl.elem2.loc16_33.2: @C.as.Copy.impl.Op.%.loc16_33.12 (%.591) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc16_33.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_33.2: <specific function> = specific_impl_function %impl.elem2.loc16_33.2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.loc16_33.3) [symbolic = %specific_impl_fn.loc16_33.2 (constants.%specific_impl_fn.03e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Copy.impl.Op.%C (%C.210)) -> out %return.param: @C.as.Copy.impl.Op.%C (%C.210) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Copy.impl.Op.%C (%C.210) = name_ref self, %self
// CHECK:STDOUT:     %t.ref: @C.as.Copy.impl.Op.%C.elem (%C.elem.731) = name_ref t, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:     %.loc16_31.1: ref @C.as.Copy.impl.Op.%T.as_type (%T.as_type.5d6) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc16_31.2: @C.as.Copy.impl.Op.%T.as_type (%T.as_type.5d6) = acquire_value %.loc16_31.1
// CHECK:STDOUT:     %.loc16_33.1: @C.as.Copy.impl.Op.%struct_type.t (%struct_type.t.87b) = struct_literal (%.loc16_31.2)
// CHECK:STDOUT:     %impl.elem0.loc16_31.1: @C.as.Copy.impl.Op.%.loc16_31.5 (%.81a) = impl_witness_access constants.%Copy.lookup_impl_witness.fd8, element0 [symbolic = %impl.elem0.loc16_31.2 (constants.%impl.elem0.35e)]
// CHECK:STDOUT:     %bound_method.loc16_31.1: <bound method> = bound_method %.loc16_31.2, %impl.elem0.loc16_31.1
// CHECK:STDOUT:     %Copy.facet.loc16_31.1: %Copy.type = facet_value constants.%T.as_type.5d6, (constants.%Copy.lookup_impl_witness.fd8) [symbolic = %Copy.facet.loc16_31.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %.loc16_31.3: %Copy.type = converted constants.%T.as_type.5d6, %Copy.facet.loc16_31.1 [symbolic = %Copy.facet.loc16_31.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %Copy.facet.loc16_31.2: %Copy.type = facet_value constants.%T.as_type.5d6, (constants.%Copy.lookup_impl_witness.fd8) [symbolic = %Copy.facet.loc16_31.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %.loc16_31.4: %Copy.type = converted constants.%T.as_type.5d6, %Copy.facet.loc16_31.2 [symbolic = %Copy.facet.loc16_31.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_31.1: <specific function> = specific_impl_function %impl.elem0.loc16_31.1, @Copy.WithSelf.Op(constants.%Copy.facet.40c) [symbolic = %specific_impl_fn.loc16_31.2 (constants.%specific_impl_fn.f62)]
// CHECK:STDOUT:     %bound_method.loc16_31.2: <bound method> = bound_method %.loc16_31.2, %specific_impl_fn.loc16_31.1
// CHECK:STDOUT:     %.loc16_33.2: ref @C.as.Copy.impl.Op.%T.as_type (%T.as_type.5d6) = splice_block %.loc16_33.11 {
// CHECK:STDOUT:       %.loc16_33.10: ref @C.as.Copy.impl.Op.%C (%C.210) = temporary_storage
// CHECK:STDOUT:       %.loc16_33.11: ref @C.as.Copy.impl.Op.%T.as_type (%T.as_type.5d6) = class_element_access %.loc16_33.10, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc16: init @C.as.Copy.impl.Op.%T.as_type (%T.as_type.5d6) to %.loc16_33.2 = call %bound_method.loc16_31.2(%.loc16_31.2)
// CHECK:STDOUT:     %.loc16_33.3: init @C.as.Copy.impl.Op.%T.as_type (%T.as_type.5d6) to %.loc16_33.11 = in_place_init %Copy.WithSelf.Op.call.loc16
// CHECK:STDOUT:     %.loc16_33.4: init @C.as.Copy.impl.Op.%C (%C.210) to %.loc16_33.10 = class_init (%.loc16_33.3)
// CHECK:STDOUT:     %.loc16_33.5: init @C.as.Copy.impl.Op.%C (%C.210) = converted %.loc16_33.1, %.loc16_33.4
// CHECK:STDOUT:     %.loc16_33.6: ref @C.as.Copy.impl.Op.%C (%C.210) = temporary %.loc16_33.10, %.loc16_33.5
// CHECK:STDOUT:     %.loc16_33.7: @C.as.Copy.impl.Op.%C (%C.210) = acquire_value %.loc16_33.6
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %Self.ref.loc16 [symbolic = %C (constants.%C.210)] {
// CHECK:STDOUT:       %.loc16_14.2: type = specific_constant constants.%C.210, @C(constants.%T.dc4) [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:       %Self.ref.loc16: type = name_ref Self, %.loc16_14.2 [symbolic = %C (constants.%C.210)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @C.as.Copy.impl.Op.%C (%C.210) = wrapper_binding c, %.loc16_33.7
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt.loc16_12.1: @C.as.Copy.impl.Op.%pattern_type (%pattern_type.0b5) = value_binding_pattern c [symbolic = %c.patt.loc16_12.2 (constants.%c.patt.6d2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.ref: @C.as.Copy.impl.Op.%C (%C.210) = name_ref c, %c
// CHECK:STDOUT:     %impl.elem0.loc17_14.1: @C.as.Copy.impl.Op.%.loc17_14.4 (%.df2) = impl_witness_access constants.%Copy.lookup_impl_witness.006, element0 [symbolic = %impl.elem0.loc17_14.2 (constants.%impl.elem0.83f)]
// CHECK:STDOUT:     %bound_method.loc17_14.1: <bound method> = bound_method %c.ref, %impl.elem0.loc17_14.1
// CHECK:STDOUT:     %Copy.facet.loc17_14.1: %Copy.type = facet_value constants.%C.210, (constants.%Copy.lookup_impl_witness.006) [symbolic = %Copy.facet.loc17_14.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:     %.loc17_14.1: %Copy.type = converted constants.%C.210, %Copy.facet.loc17_14.1 [symbolic = %Copy.facet.loc17_14.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:     %Copy.facet.loc17_14.2: %Copy.type = facet_value constants.%C.210, (constants.%Copy.lookup_impl_witness.006) [symbolic = %Copy.facet.loc17_14.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:     %.loc17_14.2: %Copy.type = converted constants.%C.210, %Copy.facet.loc17_14.2 [symbolic = %Copy.facet.loc17_14.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:     %specific_impl_fn.loc17_14.1: <specific function> = specific_impl_function %impl.elem0.loc17_14.1, @Copy.WithSelf.Op(constants.%Copy.facet.c47) [symbolic = %specific_impl_fn.loc17_14.2 (constants.%specific_impl_fn.f64)]
// CHECK:STDOUT:     %bound_method.loc17_14.2: <bound method> = bound_method %c.ref, %specific_impl_fn.loc17_14.1
// CHECK:STDOUT:     %.loc15_26.1: ref @C.as.Copy.impl.Op.%C (%C.210) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc17: init @C.as.Copy.impl.Op.%C (%C.210) to %.loc15_26.1 = call %bound_method.loc17_14.2(%c.ref)
// CHECK:STDOUT:     %impl.elem2.loc16_33.1: @C.as.Copy.impl.Op.%.loc16_33.12 (%.591) = impl_witness_access constants.%Destroy.lookup_impl_witness.f09, element2 [symbolic = %impl.elem2.loc16_33.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc16_33.1: <bound method> = bound_method %.loc16_33.6, %impl.elem2.loc16_33.1
// CHECK:STDOUT:     %Destroy.facet.loc16_33.1: %Destroy.type = facet_value constants.%C.210, (constants.%Destroy.lookup_impl_witness.f09) [symbolic = %Destroy.facet.loc16_33.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:     %.loc16_33.8: %Destroy.type = converted constants.%C.210, %Destroy.facet.loc16_33.1 [symbolic = %Destroy.facet.loc16_33.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:     %Destroy.facet.loc16_33.2: %Destroy.type = facet_value constants.%C.210, (constants.%Destroy.lookup_impl_witness.f09) [symbolic = %Destroy.facet.loc16_33.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:     %.loc16_33.9: %Destroy.type = converted constants.%C.210, %Destroy.facet.loc16_33.2 [symbolic = %Destroy.facet.loc16_33.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_33.1: <specific function> = specific_impl_function %impl.elem2.loc16_33.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Destroy.facet.dc5) [symbolic = %specific_impl_fn.loc16_33.2 (constants.%specific_impl_fn.03e)]
// CHECK:STDOUT:     %bound_method.loc16_33.2: <bound method> = bound_method %.loc16_33.6, %specific_impl_fn.loc16_33.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc16_33.2(%.loc16_33.6)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call.loc17 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeC(%T.loc21_11.2: %facet_type) {
// CHECK:STDOUT:   %T.patt.loc21_11.2: %pattern_type.143 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_11.2 (constants.%T.patt.256)]
// CHECK:STDOUT:   %T.loc21_11.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc21_11.1 (constants.%T.dc4)]
// CHECK:STDOUT:   %T.as_type.loc21_42.1: type = facet_access_type %T.loc21_11.1 [symbolic = %T.as_type.loc21_42.1 (constants.%T.as_type.5d6)]
// CHECK:STDOUT:   %pattern_type.loc21_40: type = pattern_type %T.as_type.loc21_42.1 [symbolic = %pattern_type.loc21_40 (constants.%pattern_type.0e9)]
// CHECK:STDOUT:   %t.param_patt.loc21_40.2: @MakeC.%pattern_type.loc21_40 (%pattern_type.0e9) = value_param_pattern [symbolic = %t.param_patt.loc21_40.2 (constants.%t.param_patt.081)]
// CHECK:STDOUT:   %t.patt.loc21_40.2: @MakeC.%pattern_type.loc21_40 (%pattern_type.0e9) = wrapper_binding_pattern t, %t.param_patt.loc21_40.2 [symbolic = %t.patt.loc21_40.2 (constants.%t.patt.4d1)]
// CHECK:STDOUT:   %C.loc21_51.1: type = class_type @C, @C(%T.loc21_11.1) [symbolic = %C.loc21_51.1 (constants.%C.210)]
// CHECK:STDOUT:   %.loc21_51.2: Core.Form = init_form %C.loc21_51.1 [symbolic = %.loc21_51.2 (constants.%.501)]
// CHECK:STDOUT:   %pattern_type.loc21_51: type = pattern_type %C.loc21_51.1 [symbolic = %pattern_type.loc21_51 (constants.%pattern_type.0b5)]
// CHECK:STDOUT:   %return.param_patt.loc21_51.2: @MakeC.%pattern_type.loc21_51 (%pattern_type.0b5) = out_param_pattern [symbolic = %return.param_patt.loc21_51.2 (constants.%return.param_patt.2c6)]
// CHECK:STDOUT:   %return.patt.loc21_45.2: @MakeC.%pattern_type.loc21_51 (%pattern_type.0b5) = return_slot_pattern %return.param_patt.loc21_51.2, %C.loc21_51.1 [symbolic = %return.patt.loc21_45.2 (constants.%return.patt.a82)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21: <witness> = require_complete_type %T.as_type.loc21_42.1 [symbolic = %require_complete.loc21 (constants.%require_complete.091)]
// CHECK:STDOUT:   %require_complete.loc22: <witness> = require_complete_type %C.loc21_51.1 [symbolic = %require_complete.loc22 (constants.%require_complete.bab)]
// CHECK:STDOUT:   %c.patt.loc22_8.2: @MakeC.%pattern_type.loc21_51 (%pattern_type.0b5) = value_binding_pattern c [symbolic = %c.patt.loc22_8.2 (constants.%c.patt.6d2)]
// CHECK:STDOUT:   %struct_type.t: type = struct_type {.t: @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6)} [symbolic = %struct_type.t (constants.%struct_type.t.87b)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc22: <witness> = lookup_impl_witness %T.loc21_11.1, @Copy [symbolic = %Copy.lookup_impl_witness.loc22 (constants.%Copy.lookup_impl_witness.fd8)]
// CHECK:STDOUT:   %Copy.facet.loc22_23.3: %Copy.type = facet_value %T.as_type.loc21_42.1, (%Copy.lookup_impl_witness.loc22) [symbolic = %Copy.facet.loc22_23.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc22: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc22_23.3) [symbolic = %Copy.WithSelf.Op.type.loc22 (constants.%Copy.WithSelf.Op.type.f43)]
// CHECK:STDOUT:   %.loc22_23.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type.loc22, %Copy.facet.loc22_23.3 [symbolic = %.loc22_23.3 (constants.%.81a)]
// CHECK:STDOUT:   %impl.elem0.loc22_23.2: @MakeC.%.loc22_23.3 (%.81a) = impl_witness_access %Copy.lookup_impl_witness.loc22, element0 [symbolic = %impl.elem0.loc22_23.2 (constants.%impl.elem0.35e)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_23.2: <specific function> = specific_impl_function %impl.elem0.loc22_23.2, @Copy.WithSelf.Op(%Copy.facet.loc22_23.3) [symbolic = %specific_impl_fn.loc22_23.2 (constants.%specific_impl_fn.f62)]
// CHECK:STDOUT:   %.loc23_10.3: require_specific_def_type = require_specific_def @C.as.Copy.impl(%T.loc21_11.1) [symbolic = %.loc23_10.3 (constants.%.e8c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc23: <witness> = lookup_impl_witness %C.loc21_51.1, @Copy [symbolic = %Copy.lookup_impl_witness.loc23 (constants.%Copy.lookup_impl_witness.006)]
// CHECK:STDOUT:   %Copy.facet.loc23_10.3: %Copy.type = facet_value %C.loc21_51.1, (%Copy.lookup_impl_witness.loc23) [symbolic = %Copy.facet.loc23_10.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc23: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc23_10.3) [symbolic = %Copy.WithSelf.Op.type.loc23 (constants.%Copy.WithSelf.Op.type.82d)]
// CHECK:STDOUT:   %.loc23_10.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.loc23, %Copy.facet.loc23_10.3 [symbolic = %.loc23_10.4 (constants.%.df2)]
// CHECK:STDOUT:   %impl.elem0.loc23_10.2: @MakeC.%.loc23_10.4 (%.df2) = impl_witness_access %Copy.lookup_impl_witness.loc23, element0 [symbolic = %impl.elem0.loc23_10.2 (constants.%impl.elem0.83f)]
// CHECK:STDOUT:   %specific_impl_fn.loc23_10.2: <specific function> = specific_impl_function %impl.elem0.loc23_10.2, @Copy.WithSelf.Op(%Copy.facet.loc23_10.3) [symbolic = %specific_impl_fn.loc23_10.2 (constants.%specific_impl_fn.f64)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc21_51.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.f09)]
// CHECK:STDOUT:   %Destroy.facet.loc22_24.3: %Destroy.type = facet_value %C.loc21_51.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc22_24.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.loc22_24.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.7ac)]
// CHECK:STDOUT:   %.loc22_24.12: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.loc22_24.3 [symbolic = %.loc22_24.12 (constants.%.591)]
// CHECK:STDOUT:   %impl.elem2.loc22_24.2: @MakeC.%.loc22_24.12 (%.591) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc22_24.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_24.2: <specific function> = specific_impl_function %impl.elem2.loc22_24.2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.loc22_24.3) [symbolic = %specific_impl_fn.loc22_24.2 (constants.%specific_impl_fn.03e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6)) -> out %return.param: @MakeC.%C.loc21_51.1 (%C.210) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6) = name_ref t, %t
// CHECK:STDOUT:     %.loc22_24.1: @MakeC.%struct_type.t (%struct_type.t.87b) = struct_literal (%t.ref)
// CHECK:STDOUT:     %impl.elem0.loc22_23.1: @MakeC.%.loc22_23.3 (%.81a) = impl_witness_access constants.%Copy.lookup_impl_witness.fd8, element0 [symbolic = %impl.elem0.loc22_23.2 (constants.%impl.elem0.35e)]
// CHECK:STDOUT:     %bound_method.loc22_23.1: <bound method> = bound_method %t.ref, %impl.elem0.loc22_23.1
// CHECK:STDOUT:     %Copy.facet.loc22_23.1: %Copy.type = facet_value constants.%T.as_type.5d6, (constants.%Copy.lookup_impl_witness.fd8) [symbolic = %Copy.facet.loc22_23.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %.loc22_23.1: %Copy.type = converted constants.%T.as_type.5d6, %Copy.facet.loc22_23.1 [symbolic = %Copy.facet.loc22_23.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %Copy.facet.loc22_23.2: %Copy.type = facet_value constants.%T.as_type.5d6, (constants.%Copy.lookup_impl_witness.fd8) [symbolic = %Copy.facet.loc22_23.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %.loc22_23.2: %Copy.type = converted constants.%T.as_type.5d6, %Copy.facet.loc22_23.2 [symbolic = %Copy.facet.loc22_23.3 (constants.%Copy.facet.40c)]
// CHECK:STDOUT:     %specific_impl_fn.loc22_23.1: <specific function> = specific_impl_function %impl.elem0.loc22_23.1, @Copy.WithSelf.Op(constants.%Copy.facet.40c) [symbolic = %specific_impl_fn.loc22_23.2 (constants.%specific_impl_fn.f62)]
// CHECK:STDOUT:     %bound_method.loc22_23.2: <bound method> = bound_method %t.ref, %specific_impl_fn.loc22_23.1
// CHECK:STDOUT:     %.loc22_24.2: ref @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6) = splice_block %.loc22_24.11 {
// CHECK:STDOUT:       %.loc22_24.10: ref @MakeC.%C.loc21_51.1 (%C.210) = temporary_storage
// CHECK:STDOUT:       %.loc22_24.11: ref @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6) = class_element_access %.loc22_24.10, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc22: init @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6) to %.loc22_24.2 = call %bound_method.loc22_23.2(%t.ref)
// CHECK:STDOUT:     %.loc22_24.3: init @MakeC.%T.as_type.loc21_42.1 (%T.as_type.5d6) to %.loc22_24.11 = in_place_init %Copy.WithSelf.Op.call.loc22
// CHECK:STDOUT:     %.loc22_24.4: init @MakeC.%C.loc21_51.1 (%C.210) to %.loc22_24.10 = class_init (%.loc22_24.3)
// CHECK:STDOUT:     %.loc22_24.5: init @MakeC.%C.loc21_51.1 (%C.210) = converted %.loc22_24.1, %.loc22_24.4
// CHECK:STDOUT:     %.loc22_24.6: ref @MakeC.%C.loc21_51.1 (%C.210) = temporary %.loc22_24.10, %.loc22_24.5
// CHECK:STDOUT:     %.loc22_24.7: @MakeC.%C.loc21_51.1 (%C.210) = acquire_value %.loc22_24.6
// CHECK:STDOUT:     %.loc22_13: type = splice_block %C.loc22 [symbolic = %C.loc21_51.1 (constants.%C.210)] {
// CHECK:STDOUT:       %C.ref.loc22: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref.loc22: %facet_type = name_ref T, %T.loc21_11.2 [symbolic = %T.loc21_11.1 (constants.%T.dc4)]
// CHECK:STDOUT:       %C.loc22: type = class_type @C, @C(constants.%T.dc4) [symbolic = %C.loc21_51.1 (constants.%C.210)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @MakeC.%C.loc21_51.1 (%C.210) = wrapper_binding c, %.loc22_24.7
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt.loc22_8.1: @MakeC.%pattern_type.loc21_51 (%pattern_type.0b5) = value_binding_pattern c [symbolic = %c.patt.loc22_8.2 (constants.%c.patt.6d2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.ref: @MakeC.%C.loc21_51.1 (%C.210) = name_ref c, %c
// CHECK:STDOUT:     %impl.elem0.loc23_10.1: @MakeC.%.loc23_10.4 (%.df2) = impl_witness_access constants.%Copy.lookup_impl_witness.006, element0 [symbolic = %impl.elem0.loc23_10.2 (constants.%impl.elem0.83f)]
// CHECK:STDOUT:     %bound_method.loc23_10.1: <bound method> = bound_method %c.ref, %impl.elem0.loc23_10.1
// CHECK:STDOUT:     %Copy.facet.loc23_10.1: %Copy.type = facet_value constants.%C.210, (constants.%Copy.lookup_impl_witness.006) [symbolic = %Copy.facet.loc23_10.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:     %.loc23_10.1: %Copy.type = converted constants.%C.210, %Copy.facet.loc23_10.1 [symbolic = %Copy.facet.loc23_10.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:     %Copy.facet.loc23_10.2: %Copy.type = facet_value constants.%C.210, (constants.%Copy.lookup_impl_witness.006) [symbolic = %Copy.facet.loc23_10.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:     %.loc23_10.2: %Copy.type = converted constants.%C.210, %Copy.facet.loc23_10.2 [symbolic = %Copy.facet.loc23_10.3 (constants.%Copy.facet.c47)]
// CHECK:STDOUT:     %specific_impl_fn.loc23_10.1: <specific function> = specific_impl_function %impl.elem0.loc23_10.1, @Copy.WithSelf.Op(constants.%Copy.facet.c47) [symbolic = %specific_impl_fn.loc23_10.2 (constants.%specific_impl_fn.f64)]
// CHECK:STDOUT:     %bound_method.loc23_10.2: <bound method> = bound_method %c.ref, %specific_impl_fn.loc23_10.1
// CHECK:STDOUT:     %.loc21_51.1: ref @MakeC.%C.loc21_51.1 (%C.210) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc23: init @MakeC.%C.loc21_51.1 (%C.210) to %.loc21_51.1 = call %bound_method.loc23_10.2(%c.ref)
// CHECK:STDOUT:     %impl.elem2.loc22_24.1: @MakeC.%.loc22_24.12 (%.591) = impl_witness_access constants.%Destroy.lookup_impl_witness.f09, element2 [symbolic = %impl.elem2.loc22_24.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc22_24.1: <bound method> = bound_method %.loc22_24.6, %impl.elem2.loc22_24.1
// CHECK:STDOUT:     %Destroy.facet.loc22_24.1: %Destroy.type = facet_value constants.%C.210, (constants.%Destroy.lookup_impl_witness.f09) [symbolic = %Destroy.facet.loc22_24.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:     %.loc22_24.8: %Destroy.type = converted constants.%C.210, %Destroy.facet.loc22_24.1 [symbolic = %Destroy.facet.loc22_24.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:     %Destroy.facet.loc22_24.2: %Destroy.type = facet_value constants.%C.210, (constants.%Destroy.lookup_impl_witness.f09) [symbolic = %Destroy.facet.loc22_24.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:     %.loc22_24.9: %Destroy.type = converted constants.%C.210, %Destroy.facet.loc22_24.2 [symbolic = %Destroy.facet.loc22_24.3 (constants.%Destroy.facet.dc5)]
// CHECK:STDOUT:     %specific_impl_fn.loc22_24.1: <specific function> = specific_impl_function %impl.elem2.loc22_24.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Destroy.facet.dc5) [symbolic = %specific_impl_fn.loc22_24.2 (constants.%specific_impl_fn.03e)]
// CHECK:STDOUT:     %bound_method.loc22_24.2: <bound method> = bound_method %.loc22_24.6, %specific_impl_fn.loc22_24.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc22_24.2(%.loc22_24.6)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call.loc23 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeD__carbon_thunk(%_.param: ref %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeD__carbon_thunk.ref: %MakeD.type = name_ref MakeD__carbon_thunk, file.%MakeD.decl [concrete = constants.%MakeD]
// CHECK:STDOUT:   %.loc9: ref %D = splice_block %_.param {}
// CHECK:STDOUT:   %MakeD.call: init %D to %.loc9 = call %MakeD__carbon_thunk.ref()
// CHECK:STDOUT:   assign %_.param, %MakeD.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.__destroy_thunk(%self.param: ref %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %self.param, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc5_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc5_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc5_9.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc5_9.2(%self.param: ref %D) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc5_9.2(%self.param: ref %D) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc5_9.2(%self.param: ref %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeC__carbon_thunk.6aaf27387957c53b(%_.param.loc21_53.1: ref %D, %_.param.loc21_53.2: ref %C.dac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeC__carbon_thunk.6aaf27387957c53b.ref: %MakeC.type = name_ref MakeC__carbon_thunk.6aaf27387957c53b, file.%MakeC.decl [concrete = constants.%MakeC]
// CHECK:STDOUT:   %facet_value.loc21_53.1: %facet_type = facet_value constants.%D, (constants.%Copy.impl_witness.548, constants.%custom_witness.f8f19d.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc21_53.1: %facet_type = converted constants.%D, %facet_value.loc21_53.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc21_53.2: %facet_type = facet_value constants.%D, (constants.%Copy.impl_witness.548, constants.%custom_witness.f8f19d.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc21_53.2: %facet_type = converted constants.%D, %facet_value.loc21_53.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %MakeC.specific_fn: <specific function> = specific_function %MakeC__carbon_thunk.6aaf27387957c53b.ref, @MakeC(constants.%facet_value) [concrete = constants.%MakeC.specific_fn]
// CHECK:STDOUT:   %.loc21_53.3: ref %C.dac = splice_block %_.param.loc21_53.2 {}
// CHECK:STDOUT:   %.loc21_53.4: %D = acquire_value %_.param.loc21_53.1
// CHECK:STDOUT:   %MakeC.call: init %C.dac to %.loc21_53.3 = call %MakeC.specific_fn(%.loc21_53.4)
// CHECK:STDOUT:   assign %_.param.loc21_53.2, %MakeC.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_24.1(%self.param: ref %struct_type.t.9ba) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22: ref %D = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc22, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_24.1(%self.param: ref %struct_type.t.9ba) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_24.1(%self.param: ref %struct_type.t.9ba) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_24.2(%self.param: ref %C.dac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: %C.elem.54d = name_ref t, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:   %.loc22: ref %D = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc22, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_24.2(%self.param: ref %C.dac) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_24.2(%self.param: ref %C.dac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.dc4) {
// CHECK:STDOUT:   %T.patt.loc11_10.2 => constants.%T.patt.256
// CHECK:STDOUT:   %T.loc11_10.1 => constants.%T.dc4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_10.2 => constants.%T.as_type.5d6
// CHECK:STDOUT:   %require_complete => constants.%require_complete.091
// CHECK:STDOUT:   %C => constants.%C.210
// CHECK:STDOUT:   %C.elem => constants.%C.elem.731
// CHECK:STDOUT:   %struct_type.t => constants.%struct_type.t.87b
// CHECK:STDOUT:   %complete_type.loc20_1.2 => constants.%complete_type.856
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Copy.impl(constants.%T.dc4) {
// CHECK:STDOUT:   %T => constants.%T.dc4
// CHECK:STDOUT:   %C => constants.%C.210
// CHECK:STDOUT:   %.loc14 => constants.%.8ff
// CHECK:STDOUT:   %Copy.impl_witness.loc14_21.2 => constants.%Copy.impl_witness.8ad
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type => constants.%C.as.Copy.impl.Op.type.c14
// CHECK:STDOUT:   %C.as.Copy.impl.Op => constants.%C.as.Copy.impl.Op.4d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Copy.impl.Op(constants.%T.dc4) {
// CHECK:STDOUT:   %T => constants.%T.dc4
// CHECK:STDOUT:   %C => constants.%C.210
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0b5
// CHECK:STDOUT:   %self.param_patt.loc15_15.2 => constants.%self.param_patt.27c
// CHECK:STDOUT:   %self.patt.loc15_15.2 => constants.%self.patt.fb3
// CHECK:STDOUT:   %.loc15_26.2 => constants.%.501
// CHECK:STDOUT:   %return.param_patt.loc15_26.2 => constants.%return.param_patt.2c6
// CHECK:STDOUT:   %return.patt.loc15_23.2 => constants.%return.patt.a82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeC(constants.%T.dc4) {
// CHECK:STDOUT:   %T.patt.loc21_11.2 => constants.%T.patt.256
// CHECK:STDOUT:   %T.loc21_11.1 => constants.%T.dc4
// CHECK:STDOUT:   %T.as_type.loc21_42.1 => constants.%T.as_type.5d6
// CHECK:STDOUT:   %pattern_type.loc21_40 => constants.%pattern_type.0e9
// CHECK:STDOUT:   %t.param_patt.loc21_40.2 => constants.%t.param_patt.081
// CHECK:STDOUT:   %t.patt.loc21_40.2 => constants.%t.patt.4d1
// CHECK:STDOUT:   %C.loc21_51.1 => constants.%C.210
// CHECK:STDOUT:   %.loc21_51.2 => constants.%.501
// CHECK:STDOUT:   %pattern_type.loc21_51 => constants.%pattern_type.0b5
// CHECK:STDOUT:   %return.param_patt.loc21_51.2 => constants.%return.param_patt.2c6
// CHECK:STDOUT:   %return.patt.loc21_45.2 => constants.%return.patt.a82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeC(file.%.loc21) {
// CHECK:STDOUT:   %T.patt.loc21_11.2 => constants.%T.patt.256
// CHECK:STDOUT:   %T.loc21_11.1 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc21_42.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc21_40 => constants.%pattern_type.d8d
// CHECK:STDOUT:   %t.param_patt.loc21_40.2 => constants.%t.param_patt.862
// CHECK:STDOUT:   %t.patt.loc21_40.2 => constants.%t.patt.0d1
// CHECK:STDOUT:   %C.loc21_51.1 => constants.%C.dac
// CHECK:STDOUT:   %.loc21_51.2 => constants.%.01e
// CHECK:STDOUT:   %pattern_type.loc21_51 => constants.%pattern_type.1ce
// CHECK:STDOUT:   %return.param_patt.loc21_51.2 => constants.%return.param_patt.f26
// CHECK:STDOUT:   %return.patt.loc21_45.2 => constants.%return.patt.52d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%facet_value) {
// CHECK:STDOUT:   %T.patt.loc11_10.2 => constants.%T.patt.256
// CHECK:STDOUT:   %T.loc11_10.1 => constants.%facet_value
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_10.2 => constants.%D
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %C => constants.%C.dac
// CHECK:STDOUT:   %C.elem => constants.%C.elem.54d
// CHECK:STDOUT:   %struct_type.t => constants.%struct_type.t.9ba
// CHECK:STDOUT:   %complete_type.loc20_1.2 => constants.%complete_type.b13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeC(constants.%facet_value) {
// CHECK:STDOUT:   %T.patt.loc21_11.2 => constants.%T.patt.256
// CHECK:STDOUT:   %T.loc21_11.1 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc21_42.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc21_40 => constants.%pattern_type.d8d
// CHECK:STDOUT:   %t.param_patt.loc21_40.2 => constants.%t.param_patt.862
// CHECK:STDOUT:   %t.patt.loc21_40.2 => constants.%t.patt.0d1
// CHECK:STDOUT:   %C.loc21_51.1 => constants.%C.dac
// CHECK:STDOUT:   %.loc21_51.2 => constants.%.01e
// CHECK:STDOUT:   %pattern_type.loc21_51 => constants.%pattern_type.1ce
// CHECK:STDOUT:   %return.param_patt.loc21_51.2 => constants.%return.param_patt.f26
// CHECK:STDOUT:   %return.patt.loc21_45.2 => constants.%return.patt.52d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.b13
// CHECK:STDOUT:   %c.patt.loc22_8.2 => constants.%c.patt.50c
// CHECK:STDOUT:   %struct_type.t => constants.%struct_type.t.9ba
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc22 => constants.%Copy.impl_witness.548
// CHECK:STDOUT:   %Copy.facet.loc22_23.3 => constants.%Copy.facet.48d
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc22 => constants.%Copy.WithSelf.Op.type.5bb
// CHECK:STDOUT:   %.loc22_23.3 => constants.%.e9c
// CHECK:STDOUT:   %impl.elem0.loc22_23.2 => constants.%D.as.Copy.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc22_23.2 => constants.%D.as.Copy.impl.Op
// CHECK:STDOUT:   %.loc23_10.3 => constants.%.50c
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc23 => constants.%Copy.impl_witness.a7d
// CHECK:STDOUT:   %Copy.facet.loc23_10.3 => constants.%Copy.facet.569
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc23 => constants.%Copy.WithSelf.Op.type.6a6
// CHECK:STDOUT:   %.loc23_10.4 => constants.%.0a2
// CHECK:STDOUT:   %impl.elem0.loc23_10.2 => constants.%C.as.Copy.impl.Op.2ba
// CHECK:STDOUT:   %specific_impl_fn.loc23_10.2 => constants.%C.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.4
// CHECK:STDOUT:   %Destroy.facet.loc22_24.3 => constants.%Destroy.facet.d0b
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.70e
// CHECK:STDOUT:   %.loc22_24.12 => constants.%.c6c
// CHECK:STDOUT:   %impl.elem2.loc22_24.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc22_24.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Copy.impl(constants.%facet_value) {
// CHECK:STDOUT:   %T => constants.%facet_value
// CHECK:STDOUT:   %C => constants.%C.dac
// CHECK:STDOUT:   %.loc14 => constants.%.ede
// CHECK:STDOUT:   %Copy.impl_witness.loc14_21.2 => constants.%Copy.impl_witness.a7d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type => constants.%C.as.Copy.impl.Op.type.de5
// CHECK:STDOUT:   %C.as.Copy.impl.Op => constants.%C.as.Copy.impl.Op.2ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Copy.impl.Op(constants.%facet_value) {
// CHECK:STDOUT:   %T => constants.%facet_value
// CHECK:STDOUT:   %C => constants.%C.dac
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1ce
// CHECK:STDOUT:   %self.param_patt.loc15_15.2 => constants.%self.param_patt.9bb
// CHECK:STDOUT:   %self.patt.loc15_15.2 => constants.%self.patt.89a
// CHECK:STDOUT:   %.loc15_26.2 => constants.%.01e
// CHECK:STDOUT:   %return.param_patt.loc15_26.2 => constants.%return.param_patt.f26
// CHECK:STDOUT:   %return.patt.loc15_23.2 => constants.%return.patt.52d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15 => constants.%complete_type.b13
// CHECK:STDOUT:   %c.patt.loc16_12.2 => constants.%c.patt.50c
// CHECK:STDOUT:   %T.as_type => constants.%D
// CHECK:STDOUT:   %C.elem => constants.%C.elem.54d
// CHECK:STDOUT:   %require_complete.loc16 => constants.%complete_type.357
// CHECK:STDOUT:   %struct_type.t => constants.%struct_type.t.9ba
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc16 => constants.%Copy.impl_witness.548
// CHECK:STDOUT:   %Copy.facet.loc16_31.3 => constants.%Copy.facet.48d
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc16 => constants.%Copy.WithSelf.Op.type.5bb
// CHECK:STDOUT:   %.loc16_31.5 => constants.%.e9c
// CHECK:STDOUT:   %impl.elem0.loc16_31.2 => constants.%D.as.Copy.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc16_31.2 => constants.%D.as.Copy.impl.Op
// CHECK:STDOUT:   %.loc17_14.3 => constants.%.50c
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc17 => constants.%Copy.impl_witness.a7d
// CHECK:STDOUT:   %Copy.facet.loc17_14.3 => constants.%Copy.facet.569
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc17 => constants.%Copy.WithSelf.Op.type.6a6
// CHECK:STDOUT:   %.loc17_14.4 => constants.%.0a2
// CHECK:STDOUT:   %impl.elem0.loc17_14.2 => constants.%C.as.Copy.impl.Op.2ba
// CHECK:STDOUT:   %specific_impl_fn.loc17_14.2 => constants.%C.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.4
// CHECK:STDOUT:   %Destroy.facet.loc16_33.3 => constants.%Destroy.facet.d0b
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.70e
// CHECK:STDOUT:   %.loc16_33.12 => constants.%.c6c
// CHECK:STDOUT:   %impl.elem2.loc16_33.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc16_33.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_type_impls_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %pattern_type.c60: type = pattern_type %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %self.param_patt.a33: %pattern_type.c60 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.85e: %pattern_type.c60 = wrapper_binding_pattern self, %self.param_patt.a33 [symbolic]
// CHECK:STDOUT:   %.8fc: Core.Form = init_form %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %return.param_patt.434: %pattern_type.c60 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.44a: %pattern_type.c60 = return_slot_pattern %return.param_patt.434, %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.bce: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.228: %I.WithSelf.Doit.type.bce = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Doit.decl [concrete]
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
// CHECK:STDOUT:   %I.WithSelf.Doit.type.0d0: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%I.facet.f31) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.396: %I.WithSelf.Doit.type.0d0 = struct_value () [concrete]
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
// CHECK:STDOUT:   %I.WithSelf.Doit.type.eaa: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%I.facet.a68) [concrete]
// CHECK:STDOUT:   %I.WithSelf.Doit.1f6: %I.WithSelf.Doit.type.eaa = struct_value () [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
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
// CHECK:STDOUT:   %require_complete.27c: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.type.a1c: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Doit.a4e: %I.WithSelf.Doit.type.a1c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.7ed: type = fn_type_with_self_type %I.WithSelf.Doit.type.a1c, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0.fa2: %.7ed = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.58e: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b4c: %pattern_type.b3a = wrapper_binding_pattern self, %self.param_patt.58e [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1f8: <specific function> = specific_impl_function %impl.elem0.fa2, @I.WithSelf.Doit(%T) [symbolic]
// CHECK:STDOUT:   %self.param_patt.039: %pattern_type.e39 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.535: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt.039 [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc16_9.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc16_9.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc16_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc16_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc16_9.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.8fb: %pattern_type.9ef = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e64: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt.8fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %t.param_patt.169: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.490: %pattern_type.9ef = wrapper_binding_pattern t, %t.param_patt.169 [concrete]
// CHECK:STDOUT:   %F.specific_fn.45a: <specific function> = specific_function %F, @F(%I.facet.f31) [concrete]
// CHECK:STDOUT:   %t.param_patt.267: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.614: %pattern_type.e39 = wrapper_binding_pattern t, %t.param_patt.267 [concrete]
// CHECK:STDOUT:   %F.specific_fn.dcc: <specific function> = specific_function %F, @F(%I.facet.a68) [concrete]
// CHECK:STDOUT:   %.856: type = fn_type_with_self_type %I.WithSelf.Doit.type.0d0, %I.facet.f31 [concrete]
// CHECK:STDOUT:   %.2bd: type = fn_type_with_self_type %I.WithSelf.Doit.type.eaa, %I.facet.a68 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc16_9.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc16_9.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc16_9.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.535]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %B = ref_param call_param0
// CHECK:STDOUT:     %self: ref %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc16_9.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.535]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %B = ref_param call_param0
// CHECK:STDOUT:     %self: ref %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%self.param_patt.8fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e64]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %self: ref %A = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%self.param_patt.8fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e64]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %self: ref %A = wrapper_binding self, %self.param
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
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
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
// CHECK:STDOUT:   %I.WithSelf.Doit.decl: @I.WithSelf.%I.WithSelf.Doit.type (%I.WithSelf.Doit.type.bce) = fn_decl @I.WithSelf.Doit [symbolic = @I.WithSelf.%I.WithSelf.Doit (constants.%I.WithSelf.Doit.228)] {
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
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.Doit.decl [concrete = constants.%assoc0]
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc24_15.1 [symbolic = %require_complete (constants.%require_complete.27c)]
// CHECK:STDOUT:   %I.WithSelf.Doit.type: type = fn_type @I.WithSelf.Doit, @I.WithSelf(%T.loc24_7.1) [symbolic = %I.WithSelf.Doit.type (constants.%I.WithSelf.Doit.type.a1c)]
// CHECK:STDOUT:   %.loc25_11: type = fn_type_with_self_type %I.WithSelf.Doit.type, %T.loc24_7.1 [symbolic = %.loc25_11 (constants.%.7ed)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc24_7.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc25_11.2: @F.%.loc25_11 (%.7ed) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_11.2 (constants.%impl.elem0.fa2)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_11.2: <specific function> = specific_impl_function %impl.elem0.loc25_11.2, @I.WithSelf.Doit(%T.loc24_7.1) [symbolic = %specific_impl_fn.loc25_11.2 (constants.%specific_impl_fn.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc24_15.1 (%T.as_type)) -> out %return.param: @F.%T.as_type.loc24_15.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc24_15.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %Doit.ref: %I.assoc_type = name_ref Doit, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc25_11.1: @F.%.loc25_11 (%.7ed) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_11.2 (constants.%impl.elem0.fa2)]
// CHECK:STDOUT:     %bound_method.loc25_11: <bound method> = bound_method %t.ref, %impl.elem0.loc25_11.1
// CHECK:STDOUT:     %.loc25_17.1: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc24_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc25_17.2: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc24_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc25_11.1: <specific function> = specific_impl_function %impl.elem0.loc25_11.1, @I.WithSelf.Doit(constants.%T) [symbolic = %specific_impl_fn.loc25_11.2 (constants.%specific_impl_fn.1f8)]
// CHECK:STDOUT:     %bound_method.loc25_17: <bound method> = bound_method %t.ref, %specific_impl_fn.loc25_11.1
// CHECK:STDOUT:     %.loc24_21.1: ref @F.%T.as_type.loc24_15.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %I.WithSelf.Doit.call: init @F.%T.as_type.loc24_15.1 (%T.as_type) to %.loc24_21.1 = call %bound_method.loc25_17(%t.ref)
// CHECK:STDOUT:     return %I.WithSelf.Doit.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.__destroy_thunk(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %self.param, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc16_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc16_9.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc16_9.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc16_9.2(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc16_9.2(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc16_9.2(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.__destroy_thunk(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %self.param, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
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
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F__carbon_thunk.26f2099a79512a5d.ref, @F(constants.%I.facet.f31) [concrete = constants.%F.specific_fn.45a]
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
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F__carbon_thunk.7a3b9f04ddf4269e.ref, @F(constants.%I.facet.a68) [concrete = constants.%F.specific_fn.dcc]
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
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.bce
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.228
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
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.0d0
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.396
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
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.eaa
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.1f6
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
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.a1c
// CHECK:STDOUT:   %I.WithSelf.Doit => constants.%I.WithSelf.Doit.a4e
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
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.0d0
// CHECK:STDOUT:   %.loc25_11 => constants.%.856
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
// CHECK:STDOUT:   %I.WithSelf.Doit.type => constants.%I.WithSelf.Doit.type.eaa
// CHECK:STDOUT:   %.loc25_11 => constants.%.2bd
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.impl_witness.b14
// CHECK:STDOUT:   %impl.elem0.loc25_11.2 => constants.%B.as.I.impl.Doit
// CHECK:STDOUT:   %specific_impl_fn.loc25_11.2 => constants.%B.as.I.impl.Doit
// CHECK:STDOUT: }
// CHECK:STDOUT:
