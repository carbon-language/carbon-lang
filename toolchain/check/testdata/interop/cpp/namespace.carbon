// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/namespace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/namespace.carbon

// ============================================================================
// Single
// ============================================================================

// --- single.h

namespace my_namespace { void foo(); }

// --- import_single.carbon

library "[[@TEST_NAME]]";

import Cpp library "single.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.my_namespace.foo();
  //@dump-sem-ir-end
}

// --- fail_import_namespace_wrong_member_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "single.h";

fn MyF() {
  // CHECK:STDERR: fail_import_namespace_wrong_member_name.carbon:[[@LINE+4]]:3: error: member name `not_foo` not found in `Cpp.my_namespace` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.my_namespace.not_foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.my_namespace.not_foo();
}

// --- fail_import_namespace_wrong_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "single.h";

fn MyF() {
  // CHECK:STDERR: fail_import_namespace_wrong_name.carbon:[[@LINE+4]]:3: error: member name `not_my_namespace` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.not_my_namespace.foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.not_my_namespace.foo();
}

// ============================================================================
// Multiple
// ============================================================================

// --- multiple.h

namespace my_namespace1 {
  void foo1();
  namespace my_namespace2 {
    void foo2();
    namespace my_namespace3 {
      void foo3();
    }
  }
}

// --- import_multiple.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.my_namespace1.foo1();
  Cpp.my_namespace1.my_namespace2.foo2();
  Cpp.my_namespace1.my_namespace2.my_namespace3.foo3();
  //@dump-sem-ir-end
}

// ============================================================================
// Inline namespace
// ============================================================================

// --- inline.h

inline namespace { namespace N { inline namespace { void foo() {} } } }

// --- import_inline.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.N.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Inline namespace triggers ambiguity
// ============================================================================

// --- inline_ambiguity.h

namespace N { void foo(); }
inline namespace { namespace N {} }

// --- fail_import_inline_ambiguity.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline_ambiguity.h";

fn MyF() {
  // Artificially create an error message here so that the first error line for
  // this file (which otherwise has no source location) is associated with this
  // file split.
  // TODO: Pass a location into Clang when performing C++ name lookup so that
  // its error has a location attached.
  // CHECK:STDERR: fail_import_inline_ambiguity.carbon:[[@LINE+13]]:3: error: name `error` not found [NameNotFound]
  // CHECK:STDERR:   error;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: error: reference to 'N' is ambiguous [CppInteropParseError]
  // CHECK:STDERR: fail_import_inline_ambiguity.carbon:[[@LINE-13]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./inline_ambiguity.h:2:11: note: candidate found by name lookup is 'N' [CppInteropParseNote]
  // CHECK:STDERR:     2 | namespace N { void foo(); }
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR: fail_import_inline_ambiguity.carbon:[[@LINE-17]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./inline_ambiguity.h:3:30: note: candidate found by name lookup is '(anonymous namespace)::N' [CppInteropParseNote]
  // CHECK:STDERR:     3 | inline namespace { namespace N {} }
  // CHECK:STDERR:       |                              ^
  error;

  // CHECK:STDERR: fail_import_inline_ambiguity.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `N` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.N.foo();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  Cpp.N.foo();
}

// ============================================================================
// Inline namespace ambiguity not triggered
// ============================================================================

// --- inline_ambiguity_not_triggered.h

namespace N { struct X {}; }
auto foo() -> N::X;
inline namespace { namespace N {} }

// --- import_inline_ambiguity_not_triggered.carbon

library "[[@TEST_NAME]]";

import Cpp library "inline_ambiguity_not_triggered.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Carbon keyword name
// ============================================================================

// --- special_name.h

namespace base { void foo(); }

// --- fail_import_special_name_call_unescpaed.carbon

library "[[@TEST_NAME]]";

import Cpp library "special_name.h";

fn MyF() {
  // CHECK:STDERR: fail_import_special_name_call_unescpaed.carbon:[[@LINE+4]]:3: error: member name `base` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.base.foo();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  Cpp.base.foo();
}

// --- import_special_name_call_escpaed.carbon

library "[[@TEST_NAME]]";

import Cpp library "special_name.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.r#base.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Nested namespaces.
// ============================================================================

// --- nested.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
namespace A {
namespace B {
namespace C {
struct X { int n; };
}
}
}

struct Y {
  A::B::C::X x;
};
''';

fn Use(y: Cpp.Y) -> i32 {
  return y.x.n;
}

// CHECK:STDOUT: --- import_single.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .my_namespace = %my_namespace
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %my_namespace: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %my_namespace.ref: <namespace> = name_ref my_namespace, imports.%my_namespace [concrete = imports.%my_namespace]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo1.cpp_overload_set.type: type = cpp_overload_set_type @foo1.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo1.cpp_overload_set.value: %foo1.cpp_overload_set.type = cpp_overload_set_value @foo1.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo1.type: type = fn_type @foo1 [concrete]
// CHECK:STDOUT:   %foo1: %foo1.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo2.cpp_overload_set.type: type = cpp_overload_set_type @foo2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo2.cpp_overload_set.value: %foo2.cpp_overload_set.type = cpp_overload_set_value @foo2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo2.type: type = fn_type @foo2 [concrete]
// CHECK:STDOUT:   %foo2: %foo2.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo3.cpp_overload_set.type: type = cpp_overload_set_type @foo3.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo3.cpp_overload_set.value: %foo3.cpp_overload_set.type = cpp_overload_set_value @foo3.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo3.type: type = fn_type @foo3 [concrete]
// CHECK:STDOUT:   %foo3: %foo3.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .my_namespace1 = %my_namespace1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %my_namespace1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo1 = %foo1.cpp_overload_set.value
// CHECK:STDOUT:     .my_namespace2 = %my_namespace2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo1.cpp_overload_set.value: %foo1.cpp_overload_set.type = cpp_overload_set_value @foo1.cpp_overload_set [concrete = constants.%foo1.cpp_overload_set.value]
// CHECK:STDOUT:   %foo1.decl: %foo1.type = fn_decl @foo1 [concrete = constants.%foo1] {} {}
// CHECK:STDOUT:   %my_namespace2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo2 = %foo2.cpp_overload_set.value
// CHECK:STDOUT:     .my_namespace3 = %my_namespace3
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo2.cpp_overload_set.value: %foo2.cpp_overload_set.type = cpp_overload_set_value @foo2.cpp_overload_set [concrete = constants.%foo2.cpp_overload_set.value]
// CHECK:STDOUT:   %foo2.decl: %foo2.type = fn_decl @foo2 [concrete = constants.%foo2] {} {}
// CHECK:STDOUT:   %my_namespace3: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo3 = %foo3.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo3.cpp_overload_set.value: %foo3.cpp_overload_set.type = cpp_overload_set_value @foo3.cpp_overload_set [concrete = constants.%foo3.cpp_overload_set.value]
// CHECK:STDOUT:   %foo3.decl: %foo3.type = fn_decl @foo3 [concrete = constants.%foo3] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %my_namespace1.ref.loc8: <namespace> = name_ref my_namespace1, imports.%my_namespace1 [concrete = imports.%my_namespace1]
// CHECK:STDOUT:   %foo1.ref: %foo1.cpp_overload_set.type = name_ref foo1, imports.%foo1.cpp_overload_set.value [concrete = constants.%foo1.cpp_overload_set.value]
// CHECK:STDOUT:   %foo1.call: init %empty_tuple.type = call imports.%foo1.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %my_namespace1.ref.loc9: <namespace> = name_ref my_namespace1, imports.%my_namespace1 [concrete = imports.%my_namespace1]
// CHECK:STDOUT:   %my_namespace2.ref.loc9: <namespace> = name_ref my_namespace2, imports.%my_namespace2 [concrete = imports.%my_namespace2]
// CHECK:STDOUT:   %foo2.ref: %foo2.cpp_overload_set.type = name_ref foo2, imports.%foo2.cpp_overload_set.value [concrete = constants.%foo2.cpp_overload_set.value]
// CHECK:STDOUT:   %foo2.call: init %empty_tuple.type = call imports.%foo2.decl()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %my_namespace1.ref.loc10: <namespace> = name_ref my_namespace1, imports.%my_namespace1 [concrete = imports.%my_namespace1]
// CHECK:STDOUT:   %my_namespace2.ref.loc10: <namespace> = name_ref my_namespace2, imports.%my_namespace2 [concrete = imports.%my_namespace2]
// CHECK:STDOUT:   %my_namespace3.ref: <namespace> = name_ref my_namespace3, imports.%my_namespace3 [concrete = imports.%my_namespace3]
// CHECK:STDOUT:   %foo3.ref: %foo3.cpp_overload_set.type = name_ref foo3, imports.%foo3.cpp_overload_set.value [concrete = constants.%foo3.cpp_overload_set.value]
// CHECK:STDOUT:   %foo3.call: init %empty_tuple.type = call imports.%foo3.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_inline_ambiguity_not_triggered.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %ptr.13d: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.8b9: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.e4b: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.8b9 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_11.1: ref %X = temporary_storage
// CHECK:STDOUT:   %addr.loc8_11.1: %ptr.13d = addr_of %.loc8_11.1
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_11.1)
// CHECK:STDOUT:   %.loc8_11.2: init %X = in_place_init %foo__carbon_thunk.call, %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.3: ref %X = temporary %.loc8_11.1, %.loc8_11.2
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_11.4: %type_where = converted constants.%X, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.e4b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_11.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11.2: %ptr.13d = addr_of %.loc8_11.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_special_name_call_escpaed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .r#base = %base
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %base: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %base.ref: <namespace> = name_ref r#base, imports.%base [concrete = imports.%base]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
