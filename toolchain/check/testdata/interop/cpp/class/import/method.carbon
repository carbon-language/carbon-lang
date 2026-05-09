// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-std=c++23
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/method.carbon

// --- object_param_qualifiers.h

struct HasQualifiers {
  void plain();
  void const_this() const;
  void volatile_this() volatile;

  void ref_this() &;
  void const_ref_this() const&;

  void ref_ref_this() &&;
  void const_ref_ref_this() const&&;
};

// --- use_object_param_qualifiers.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: use_object_param_qualifiers.carbon:[[@LINE+8]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./object_param_qualifiers.h:11:8: error: 'this' argument to member function 'const_ref_ref_this' is an lvalue, but function has rvalue ref-qualifier [CppInteropParseError]
// CHECK:STDERR:    11 |   void const_ref_ref_this() const&&;
// CHECK:STDERR:       |        ^
// CHECK:STDERR: use_object_param_qualifiers.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./object_param_qualifiers.h:11:8: note: 'const_ref_ref_this' declared here [CppInteropParseNote]
// CHECK:STDERR:    11 |   void const_ref_ref_this() const&&;
// CHECK:STDERR:       |        ^
import Cpp library "object_param_qualifiers.h";

fn Make() -> Cpp.HasQualifiers;

fn F(v: Cpp.HasQualifiers, p: Cpp.HasQualifiers*) {
  //@dump-sem-ir-begin
  v.const_this();
  v.const_ref_this();
  // CHECK:STDERR: use_object_param_qualifiers.carbon:[[@LINE+12]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   v.const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: use_object_param_qualifiers.carbon:[[@LINE-12]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:10:8: error: 'this' argument to member function 'ref_ref_this' is an lvalue, but function has rvalue ref-qualifier [CppInteropParseError]
  // CHECK:STDERR:    10 |   void ref_ref_this() &&;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: use_object_param_qualifiers.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:10:8: note: 'ref_ref_this' declared here [CppInteropParseNote]
  // CHECK:STDERR:    10 |   void ref_ref_this() &&;
  // CHECK:STDERR:       |        ^
  v.const_ref_ref_this();

  p->plain();
  p->ref_this();
  p->const_this();
  p->const_ref_this();

  Make().plain();
  Make().const_this();
  Make().const_ref_this();
  // CHECK:STDERR: use_object_param_qualifiers.carbon:[[@LINE+4]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Make().ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Make().ref_ref_this();
  Make().const_ref_ref_this();
  //@dump-sem-ir-end
}

// --- fail_bad_object_param_qualifiers_by_value.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Value(v: Cpp.HasQualifiers) {
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+8]]:11: error: no matching function for call to 'plain' [CppInteropParseError]
  // CHECK:STDERR:    15 |   v.plain();
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:3:8: note: candidate function not viable: 'this' argument has type 'const HasQualifiers', but method is not marked const [CppInteropParseNote]
  // CHECK:STDERR:     3 |   void plain();
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR:
  v.plain();

  // TODO: This should remain invalid once we support `volatile`.
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+8]]:19: error: no matching function for call to 'volatile_this' [CppInteropParseError]
  // CHECK:STDERR:    26 |   v.volatile_this();
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE-17]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:5:8: note: candidate function not viable: 'this' argument has type 'const HasQualifiers', but method is not marked const [CppInteropParseNote]
  // CHECK:STDERR:     5 |   void volatile_this() volatile;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR:
  v.volatile_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+8]]:14: error: no matching function for call to 'ref_this' [CppInteropParseError]
  // CHECK:STDERR:    36 |   v.ref_this();
  // CHECK:STDERR:       |              ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE-27]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:7:8: note: candidate function not viable: 'this' argument has type 'const HasQualifiers', but method is not marked const [CppInteropParseNote]
  // CHECK:STDERR:     7 |   void ref_this() &;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR:
  v.ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+8]]:18: error: no matching function for call to 'ref_ref_this' [CppInteropParseError]
  // CHECK:STDERR:    46 |   v.ref_ref_this();
  // CHECK:STDERR:       |                  ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE-37]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:10:8: note: candidate function not viable: 'this' argument has type 'const HasQualifiers', but method is not marked const [CppInteropParseNote]
  // CHECK:STDERR:    10 |   void ref_ref_this() &&;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR:
  v.ref_ref_this();
}

// --- fail_todo_bad_object_param_qualifiers_by_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Ref(p: Cpp.HasQualifiers*) {
  // TODO: This should eventually be accepted if we support `volatile`.
  p->volatile_this();
}

// --- fail_bad_object_param_qualifiers_ref_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Ref(p: Cpp.HasQualifiers*) {
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+8]]:19: error: no matching function for call to 'ref_ref_this' [CppInteropParseError]
  // CHECK:STDERR:    15 |   p->ref_ref_this();
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:10:8: note: candidate function not viable: expects an rvalue for object argument [CppInteropParseNote]
  // CHECK:STDERR:    10 |   void ref_ref_this() &&;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR:
  p->ref_ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+8]]:25: error: no matching function for call to 'const_ref_ref_this' [CppInteropParseError]
  // CHECK:STDERR:    25 |   p->const_ref_ref_this();
  // CHECK:STDERR:       |                         ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:11:8: note: candidate function not viable: expects an rvalue for object argument [CppInteropParseNote]
  // CHECK:STDERR:    11 |   void const_ref_ref_this() const&&;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR:
  p->const_ref_ref_this();
}

// --- fail_bad_object_param_qualifiers_for_temporary.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Make() -> Cpp.HasQualifiers;

fn F() {
  // CHECK:STDERR: fail_bad_object_param_qualifiers_for_temporary.carbon:[[@LINE+8]]:19: error: no matching function for call to 'ref_this' [CppInteropParseError]
  // CHECK:STDERR:    17 |   Make().ref_this();
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_for_temporary.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./object_param_qualifiers.h:7:8: note: candidate function not viable: expects an lvalue for object argument [CppInteropParseNote]
  // CHECK:STDERR:     7 |   void ref_this() &;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR:
  Make().ref_this();
}

// --- object_param_qualifiers_overloaded.h

struct NoRefQualifier {
  int* _Nonnull F();
  int F() const;
  void F() volatile;
};

struct WithRefQualifier {
  int* _Nonnull F() &;
  int F() const &;
  void F() volatile &;
};

// --- use_object_param_qualifiers_overloaded.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers_overloaded.h";

fn CallFNoRefQualifier(v: Cpp.NoRefQualifier, p: Cpp.NoRefQualifier*) {
  //@dump-sem-ir-begin
  var unused a: i32 = v.F();
  var unused b: i32* = p->F();
  //@dump-sem-ir-end
}

fn CallFWithRefQualifier(v: Cpp.WithRefQualifier, p: Cpp.WithRefQualifier*) {
  //@dump-sem-ir-begin
  var unused a: i32 = v.F();
  var unused b: i32* = p->F();
  //@dump-sem-ir-end
}

// --- explicit_object_param.h

struct Another {
};

struct ExplicitObjectParam {
  void F(this ExplicitObjectParam);
  void G(this int);
  void H(this Another);
};

// --- call_explicit_object_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "explicit_object_param.h";

fn Call(e: Cpp.ExplicitObjectParam, n: i32, a: Cpp.Another) {
  //@dump-sem-ir-begin
  e.F();
  n.(Cpp.ExplicitObjectParam.G)();
  a.(Cpp.ExplicitObjectParam.H)();
  //@dump-sem-ir-end
}

// --- fail_call_explicit_object_param_with_unsupported_type_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct ExplicitObjectParam {
  void F(this ExplicitObjectParam, _BitInt(23) x);
};
''';

fn Call(e: Cpp.ExplicitObjectParam) {
  // CHECK:STDERR: fail_call_explicit_object_param_with_unsupported_type_param.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: parameter type: _BitInt(23)` [SemanticsTodo]
  // CHECK:STDERR:   e.(Cpp.ExplicitObjectParam.F)(1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  e.(Cpp.ExplicitObjectParam.F)(1);
}

// --- explicit_object_param_overloaded.h

struct Another {
};

struct ExplicitObjectParam {
  void F(this ExplicitObjectParam);
  void F(this int);
  void F(this Another);
};

// --- call_explicit_object_param_overloaded.carbon

library "[[@TEST_NAME]]";

import Cpp library "explicit_object_param_overloaded.h";

fn Call(e: Cpp.ExplicitObjectParam, n: i32, a: Cpp.Another) {
  //@dump-sem-ir-begin
  e.F();
  n.(Cpp.ExplicitObjectParam.F)();
  a.(Cpp.ExplicitObjectParam.F)();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- use_object_param_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasQualifiers: type = class_type @HasQualifiers [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ec3: type = ptr_type %HasQualifiers [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_this.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.const_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_this.cpp_overload_set.value: %HasQualifiers.const_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.const_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_this__carbon_thunk.type: type = fn_type @const_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %const_this__carbon_thunk: %const_this__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.const_ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.cpp_overload_set.value: %HasQualifiers.const_ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.const_ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.type: type = fn_type @const_ref_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk: %const_ref_this__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_ref_this.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.const_ref_ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_ref_this.cpp_overload_set.value: %HasQualifiers.const_ref_ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.const_ref_ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_ref_this.type: type = fn_type @HasQualifiers.const_ref_ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_ref_this: %HasQualifiers.const_ref_ref_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.plain.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.cpp_overload_set.value: %HasQualifiers.plain.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.plain.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.type: type = fn_type @HasQualifiers.plain [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain: %HasQualifiers.plain.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.cpp_overload_set.value: %HasQualifiers.ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.type: type = fn_type @HasQualifiers.ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this: %HasQualifiers.ref_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.ref_ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.cpp_overload_set.value: %HasQualifiers.ref_ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.ref_ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.type: type = fn_type @HasQualifiers.ref_ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this: %HasQualifiers.ref_ref_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.type: type = fn_type @HasQualifiers.cpp_destructor [concrete]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor: %HasQualifiers.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.Op.type: type = fn_type @HasQualifiers.Op [concrete]
// CHECK:STDOUT:   %HasQualifiers.Op: %HasQualifiers.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %HasQualifiers.const_this.cpp_overload_set.value: %HasQualifiers.const_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.const_this.cpp_overload_set [concrete = constants.%HasQualifiers.const_this.cpp_overload_set.value]
// CHECK:STDOUT:   %const_this__carbon_thunk.decl: %const_this__carbon_thunk.type = fn_decl @const_this__carbon_thunk [concrete = constants.%const_this__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.cpp_overload_set.value: %HasQualifiers.const_ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.const_ref_this.cpp_overload_set [concrete = constants.%HasQualifiers.const_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.decl: %const_ref_this__carbon_thunk.type = fn_decl @const_ref_this__carbon_thunk [concrete = constants.%const_ref_this__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.const_ref_ref_this.cpp_overload_set.value: %HasQualifiers.const_ref_ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.const_ref_ref_this.cpp_overload_set [concrete = constants.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %HasQualifiers.const_ref_ref_this.decl: %HasQualifiers.const_ref_ref_this.type = fn_decl @HasQualifiers.const_ref_ref_this [concrete = constants.%HasQualifiers.const_ref_ref_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.plain.cpp_overload_set.value: %HasQualifiers.plain.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.plain.cpp_overload_set [concrete = constants.%HasQualifiers.plain.cpp_overload_set.value]
// CHECK:STDOUT:   %HasQualifiers.plain.decl: %HasQualifiers.plain.type = fn_decl @HasQualifiers.plain [concrete = constants.%HasQualifiers.plain] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.ref_this.cpp_overload_set.value: %HasQualifiers.ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.ref_this.cpp_overload_set [concrete = constants.%HasQualifiers.ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %HasQualifiers.ref_this.decl: %HasQualifiers.ref_this.type = fn_decl @HasQualifiers.ref_this [concrete = constants.%HasQualifiers.ref_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.cpp_overload_set.value: %HasQualifiers.ref_ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.ref_ref_this.cpp_overload_set [concrete = constants.%HasQualifiers.ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.decl: %HasQualifiers.ref_ref_this.type = fn_decl @HasQualifiers.ref_ref_this [concrete = constants.%HasQualifiers.ref_ref_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.decl: %HasQualifiers.cpp_destructor.type = fn_decl @HasQualifiers.cpp_destructor [concrete = constants.%HasQualifiers.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%v.param: %HasQualifiers, %p.param: %ptr.ec3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref.loc18: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_this.ref.loc18: %HasQualifiers.const_this.cpp_overload_set.type = name_ref const_this, imports.%HasQualifiers.const_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %v.ref.loc18, %const_this.ref.loc18
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc18: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%v.ref.loc18)
// CHECK:STDOUT:   %v.ref.loc19: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_ref_this.ref.loc19: %HasQualifiers.const_ref_this.cpp_overload_set.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %v.ref.loc19, %const_ref_this.ref.loc19
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%v.ref.loc19)
// CHECK:STDOUT:   %v.ref.loc32: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_ref_ref_this.ref.loc32: %HasQualifiers.const_ref_ref_this.cpp_overload_set.type = name_ref const_ref_ref_this, imports.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc32: <bound method> = bound_method %v.ref.loc32, %const_ref_ref_this.ref.loc32
// CHECK:STDOUT:   %HasQualifiers.const_ref_ref_this.call.loc32: init %empty_tuple.type = call imports.%HasQualifiers.const_ref_ref_this.decl(%v.ref.loc32)
// CHECK:STDOUT:   %p.ref.loc34: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc34: ref %HasQualifiers = deref %p.ref.loc34
// CHECK:STDOUT:   %plain.ref.loc34: %HasQualifiers.plain.cpp_overload_set.type = name_ref plain, imports.%HasQualifiers.plain.cpp_overload_set.value [concrete = constants.%HasQualifiers.plain.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %.loc34, %plain.ref.loc34
// CHECK:STDOUT:   %HasQualifiers.plain.call.loc34: init %empty_tuple.type = call imports.%HasQualifiers.plain.decl(%.loc34)
// CHECK:STDOUT:   %p.ref.loc35: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc35: ref %HasQualifiers = deref %p.ref.loc35
// CHECK:STDOUT:   %ref_this.ref: %HasQualifiers.ref_this.cpp_overload_set.type = name_ref ref_this, imports.%HasQualifiers.ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc35: <bound method> = bound_method %.loc35, %ref_this.ref
// CHECK:STDOUT:   %HasQualifiers.ref_this.call: init %empty_tuple.type = call imports.%HasQualifiers.ref_this.decl(%.loc35)
// CHECK:STDOUT:   %p.ref.loc36: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc36_4.1: ref %HasQualifiers = deref %p.ref.loc36
// CHECK:STDOUT:   %const_this.ref.loc36: %HasQualifiers.const_this.cpp_overload_set.type = name_ref const_this, imports.%HasQualifiers.const_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc36: <bound method> = bound_method %.loc36_4.1, %const_this.ref.loc36
// CHECK:STDOUT:   %.loc36_4.2: %HasQualifiers = acquire_value %.loc36_4.1
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc36: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%.loc36_4.2)
// CHECK:STDOUT:   %p.ref.loc37: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc37_4.1: ref %HasQualifiers = deref %p.ref.loc37
// CHECK:STDOUT:   %const_ref_this.ref.loc37: %HasQualifiers.const_ref_this.cpp_overload_set.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc37: <bound method> = bound_method %.loc37_4.1, %const_ref_this.ref.loc37
// CHECK:STDOUT:   %.loc37_4.2: %HasQualifiers = acquire_value %.loc37_4.1
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc37: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%.loc37_4.2)
// CHECK:STDOUT:   %Make.ref.loc39: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc39_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc39: init %HasQualifiers to %.loc39_8.1 = call %Make.ref.loc39()
// CHECK:STDOUT:   %.loc39_8.2: ref %HasQualifiers = temporary %.loc39_8.1, %Make.call.loc39
// CHECK:STDOUT:   %plain.ref.loc39: %HasQualifiers.plain.cpp_overload_set.type = name_ref plain, imports.%HasQualifiers.plain.cpp_overload_set.value [concrete = constants.%HasQualifiers.plain.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc39: <bound method> = bound_method %.loc39_8.2, %plain.ref.loc39
// CHECK:STDOUT:   %HasQualifiers.plain.call.loc39: init %empty_tuple.type = call imports.%HasQualifiers.plain.decl(%.loc39_8.2)
// CHECK:STDOUT:   %Make.ref.loc40: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc40_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc40: init %HasQualifiers to %.loc40_8.1 = call %Make.ref.loc40()
// CHECK:STDOUT:   %.loc40_8.2: ref %HasQualifiers = temporary %.loc40_8.1, %Make.call.loc40
// CHECK:STDOUT:   %const_this.ref.loc40: %HasQualifiers.const_this.cpp_overload_set.type = name_ref const_this, imports.%HasQualifiers.const_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc40: <bound method> = bound_method %.loc40_8.2, %const_this.ref.loc40
// CHECK:STDOUT:   %.loc40_8.3: %HasQualifiers = acquire_value %.loc40_8.2
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc40: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%.loc40_8.3)
// CHECK:STDOUT:   %Make.ref.loc41: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc41_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc41: init %HasQualifiers to %.loc41_8.1 = call %Make.ref.loc41()
// CHECK:STDOUT:   %.loc41_8.2: ref %HasQualifiers = temporary %.loc41_8.1, %Make.call.loc41
// CHECK:STDOUT:   %const_ref_this.ref.loc41: %HasQualifiers.const_ref_this.cpp_overload_set.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc41: <bound method> = bound_method %.loc41_8.2, %const_ref_this.ref.loc41
// CHECK:STDOUT:   %.loc41_8.3: %HasQualifiers = acquire_value %.loc41_8.2
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc41: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%.loc41_8.3)
// CHECK:STDOUT:   %Make.ref.loc46: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc46_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc46: init %HasQualifiers to %.loc46_8.1 = call %Make.ref.loc46()
// CHECK:STDOUT:   %.loc46_8.2: ref %HasQualifiers = temporary %.loc46_8.1, %Make.call.loc46
// CHECK:STDOUT:   %ref_ref_this.ref: %HasQualifiers.ref_ref_this.cpp_overload_set.type = name_ref ref_ref_this, imports.%HasQualifiers.ref_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc46: <bound method> = bound_method %.loc46_8.2, %ref_ref_this.ref
// CHECK:STDOUT:   %.loc46_8.3: %HasQualifiers = acquire_value %.loc46_8.2
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.call: init %empty_tuple.type = call imports.%HasQualifiers.ref_ref_this.decl(%.loc46_8.3)
// CHECK:STDOUT:   %Make.ref.loc47: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc47_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc47: init %HasQualifiers to %.loc47_8.1 = call %Make.ref.loc47()
// CHECK:STDOUT:   %.loc47_8.2: ref %HasQualifiers = temporary %.loc47_8.1, %Make.call.loc47
// CHECK:STDOUT:   %const_ref_ref_this.ref.loc47: %HasQualifiers.const_ref_ref_this.cpp_overload_set.type = name_ref const_ref_ref_this, imports.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc47: <bound method> = bound_method %.loc47_8.2, %const_ref_ref_this.ref.loc47
// CHECK:STDOUT:   %.loc47_8.3: %HasQualifiers = acquire_value %.loc47_8.2
// CHECK:STDOUT:   %HasQualifiers.const_ref_ref_this.call.loc47: init %empty_tuple.type = call imports.%HasQualifiers.const_ref_ref_this.decl(%.loc47_8.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc47: <bound method> = bound_method %.loc47_8.2, constants.%HasQualifiers.Op
// CHECK:STDOUT:   %Op.ref.loc47: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc47: <bound method> = bound_method %.loc47_8.2, %Op.ref.loc47
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc47: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc47(%.loc47_8.2)
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc46: <bound method> = bound_method %.loc46_8.2, constants.%HasQualifiers.Op
// CHECK:STDOUT:   %Op.ref.loc46: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc46: <bound method> = bound_method %.loc46_8.2, %Op.ref.loc46
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc46: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc46(%.loc46_8.2)
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc41: <bound method> = bound_method %.loc41_8.2, constants.%HasQualifiers.Op
// CHECK:STDOUT:   %Op.ref.loc41: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc41: <bound method> = bound_method %.loc41_8.2, %Op.ref.loc41
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc41: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc41(%.loc41_8.2)
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc40: <bound method> = bound_method %.loc40_8.2, constants.%HasQualifiers.Op
// CHECK:STDOUT:   %Op.ref.loc40: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc40: <bound method> = bound_method %.loc40_8.2, %Op.ref.loc40
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc40: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc40(%.loc40_8.2)
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc39: <bound method> = bound_method %.loc39_8.2, constants.%HasQualifiers.Op
// CHECK:STDOUT:   %Op.ref.loc39: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc39: <bound method> = bound_method %.loc39_8.2, %Op.ref.loc39
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc39: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc39(%.loc39_8.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_object_param_qualifiers_overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoRefQualifier: type = class_type @NoRefQualifier [concrete]
// CHECK:STDOUT:   %ptr.2a2: type = ptr_type %NoRefQualifier [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %NoRefQualifier.F.cpp_overload_set.type: type = cpp_overload_set_type @NoRefQualifier.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NoRefQualifier.F.cpp_overload_set.value: %NoRefQualifier.F.cpp_overload_set.type = cpp_overload_set_value @NoRefQualifier.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.1: type = fn_type @F__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.1: %F__carbon_thunk.type.eda1ac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %NoRefQualifier.F.type.b65611.2: type = fn_type @NoRefQualifier.F.2 [concrete]
// CHECK:STDOUT:   %NoRefQualifier.F.d50a5a.2: %NoRefQualifier.F.type.b65611.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %WithRefQualifier: type = class_type @WithRefQualifier [concrete]
// CHECK:STDOUT:   %ptr.c86: type = ptr_type %WithRefQualifier [concrete]
// CHECK:STDOUT:   %WithRefQualifier.F.cpp_overload_set.type: type = cpp_overload_set_type @WithRefQualifier.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %WithRefQualifier.F.cpp_overload_set.value: %WithRefQualifier.F.cpp_overload_set.type = cpp_overload_set_value @WithRefQualifier.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.2: type = fn_type @F__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.2: %F__carbon_thunk.type.eda1ac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %WithRefQualifier.F.type.7c19e2.2: type = fn_type @WithRefQualifier.F.2 [concrete]
// CHECK:STDOUT:   %WithRefQualifier.F.d90b51.2: %WithRefQualifier.F.type.7c19e2.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %NoRefQualifier.F.cpp_overload_set.value: %NoRefQualifier.F.cpp_overload_set.type = cpp_overload_set_value @NoRefQualifier.F.cpp_overload_set [concrete = constants.%NoRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.1: %F__carbon_thunk.type.eda1ac.1 = fn_decl @F__carbon_thunk.1 [concrete = constants.%F__carbon_thunk.0cd6a8.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NoRefQualifier.F.decl.3dba03.2: %NoRefQualifier.F.type.b65611.2 = fn_decl @NoRefQualifier.F.2 [concrete = constants.%NoRefQualifier.F.d50a5a.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithRefQualifier.F.cpp_overload_set.value: %WithRefQualifier.F.cpp_overload_set.type = cpp_overload_set_value @WithRefQualifier.F.cpp_overload_set [concrete = constants.%WithRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.2: %F__carbon_thunk.type.eda1ac.2 = fn_decl @F__carbon_thunk.2 [concrete = constants.%F__carbon_thunk.0cd6a8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithRefQualifier.F.decl.e3f32b.2: %WithRefQualifier.F.type.7c19e2.2 = fn_decl @WithRefQualifier.F.2 [concrete = constants.%WithRefQualifier.F.d90b51.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFNoRefQualifier(%v.param: %NoRefQualifier, %p.param: %ptr.2a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %v.ref: %NoRefQualifier = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc8: %NoRefQualifier.F.cpp_overload_set.type = name_ref F, imports.%NoRefQualifier.F.cpp_overload_set.value [concrete = constants.%NoRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %v.ref, %F.ref.loc8
// CHECK:STDOUT:   %F__carbon_thunk.call: init %i32 = call imports.%F__carbon_thunk.decl.e1b8ec.1(%v.ref)
// CHECK:STDOUT:   assign %a.var, %F__carbon_thunk.call
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.fe8 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.fe8 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %ptr.235 = var %b.var_patt
// CHECK:STDOUT:   %p.ref: %ptr.2a2 = name_ref p, %p
// CHECK:STDOUT:   %.loc9_25: ref %NoRefQualifier = deref %p.ref
// CHECK:STDOUT:   %F.ref.loc9: %NoRefQualifier.F.cpp_overload_set.type = name_ref F, imports.%NoRefQualifier.F.cpp_overload_set.value [concrete = constants.%NoRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_25, %F.ref.loc9
// CHECK:STDOUT:   %NoRefQualifier.F.call: init %ptr.235 = call imports.%NoRefQualifier.F.decl.3dba03.2(%.loc9_25)
// CHECK:STDOUT:   assign %b.var, %NoRefQualifier.F.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %ptr.loc9 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %i32.loc9 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ptr.235 = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9(%self.param: ref %ptr.235) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFWithRefQualifier(%v.param: %WithRefQualifier, %p.param: %ptr.c86) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %v.ref: %WithRefQualifier = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc15: %WithRefQualifier.F.cpp_overload_set.type = name_ref F, imports.%WithRefQualifier.F.cpp_overload_set.value [concrete = constants.%WithRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %v.ref, %F.ref.loc15
// CHECK:STDOUT:   %F__carbon_thunk.call: init %i32 = call imports.%F__carbon_thunk.decl.e1b8ec.2(%v.ref)
// CHECK:STDOUT:   assign %a.var, %F__carbon_thunk.call
// CHECK:STDOUT:   %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.fe8 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.fe8 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %ptr.235 = var %b.var_patt
// CHECK:STDOUT:   %p.ref: %ptr.c86 = name_ref p, %p
// CHECK:STDOUT:   %.loc16_25: ref %WithRefQualifier = deref %p.ref
// CHECK:STDOUT:   %F.ref.loc16: %WithRefQualifier.F.cpp_overload_set.type = name_ref F, imports.%WithRefQualifier.F.cpp_overload_set.value [concrete = constants.%WithRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %.loc16_25, %F.ref.loc16
// CHECK:STDOUT:   %WithRefQualifier.F.call: init %ptr.235 = call imports.%WithRefQualifier.F.decl.e3f32b.2(%.loc16_25)
// CHECK:STDOUT:   assign %b.var, %WithRefQualifier.F.call
// CHECK:STDOUT:   %.loc16_20: type = splice_block %ptr.loc16 [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %i32.loc16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %i32.loc16 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ptr.235 = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc16: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc16: init %empty_tuple.type = call %Destroy.Op.bound.loc16(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc15: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc15: init %empty_tuple.type = call %Destroy.Op.bound.loc15(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_explicit_object_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ExplicitObjectParam: type = class_type @ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Another: type = class_type @Another [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.F.cpp_overload_set.type: type = cpp_overload_set_type @ExplicitObjectParam.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.F.cpp_overload_set.value: %ExplicitObjectParam.F.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.7f5: type = ptr_type %ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type: type = fn_type @F__carbon_thunk [concrete]
// CHECK:STDOUT:   %F__carbon_thunk: %F__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.G.cpp_overload_set.type: type = cpp_overload_set_type @ExplicitObjectParam.G.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.G.cpp_overload_set.value: %ExplicitObjectParam.G.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.G.cpp_overload_set [concrete]
// CHECK:STDOUT:   %G__carbon_thunk.type: type = fn_type @G__carbon_thunk [concrete]
// CHECK:STDOUT:   %G__carbon_thunk: %G__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.H.cpp_overload_set.type: type = cpp_overload_set_type @ExplicitObjectParam.H.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.H.cpp_overload_set.value: %ExplicitObjectParam.H.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.H.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.289: type = ptr_type %Another [concrete]
// CHECK:STDOUT:   %H__carbon_thunk.type: type = fn_type @H__carbon_thunk [concrete]
// CHECK:STDOUT:   %H__carbon_thunk: %H__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ExplicitObjectParam = %ExplicitObjectParam.decl
// CHECK:STDOUT:     .Another = %Another.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitObjectParam.decl: type = class_decl @ExplicitObjectParam [concrete = constants.%ExplicitObjectParam] {} {}
// CHECK:STDOUT:   %Another.decl: type = class_decl @Another [concrete = constants.%Another] {} {}
// CHECK:STDOUT:   %ExplicitObjectParam.F.cpp_overload_set.value: %ExplicitObjectParam.F.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.F.cpp_overload_set [concrete = constants.%ExplicitObjectParam.F.cpp_overload_set.value]
// CHECK:STDOUT:   %F__carbon_thunk.decl: %F__carbon_thunk.type = fn_decl @F__carbon_thunk [concrete = constants.%F__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitObjectParam.G.cpp_overload_set.value: %ExplicitObjectParam.G.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.G.cpp_overload_set [concrete = constants.%ExplicitObjectParam.G.cpp_overload_set.value]
// CHECK:STDOUT:   %G__carbon_thunk.decl: %G__carbon_thunk.type = fn_decl @G__carbon_thunk [concrete = constants.%G__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitObjectParam.H.cpp_overload_set.value: %ExplicitObjectParam.H.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.H.cpp_overload_set [concrete = constants.%ExplicitObjectParam.H.cpp_overload_set.value]
// CHECK:STDOUT:   %H__carbon_thunk.decl: %H__carbon_thunk.type = fn_decl @H__carbon_thunk [concrete = constants.%H__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%e.param: %ExplicitObjectParam, %n.param: %i32, %a.param: %Another) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %e.ref: %ExplicitObjectParam = name_ref e, %e
// CHECK:STDOUT:   %F.ref: %ExplicitObjectParam.F.cpp_overload_set.type = name_ref F, imports.%ExplicitObjectParam.F.cpp_overload_set.value [concrete = constants.%ExplicitObjectParam.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %e.ref, %F.ref
// CHECK:STDOUT:   %.loc8: ref %ExplicitObjectParam = value_as_ref %e.ref
// CHECK:STDOUT:   %addr.loc8: %ptr.7f5 = addr_of %.loc8
// CHECK:STDOUT:   %F__carbon_thunk.call: init %empty_tuple.type = call imports.%F__carbon_thunk.decl(%addr.loc8)
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc9: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %G.ref: %ExplicitObjectParam.G.cpp_overload_set.type = name_ref G, imports.%ExplicitObjectParam.G.cpp_overload_set.value [concrete = constants.%ExplicitObjectParam.G.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %n.ref, %G.ref
// CHECK:STDOUT:   %G__carbon_thunk.call: init %empty_tuple.type = call imports.%G__carbon_thunk.decl(%n.ref)
// CHECK:STDOUT:   %a.ref: %Another = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc10: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %H.ref: %ExplicitObjectParam.H.cpp_overload_set.type = name_ref H, imports.%ExplicitObjectParam.H.cpp_overload_set.value [concrete = constants.%ExplicitObjectParam.H.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref, %H.ref
// CHECK:STDOUT:   %.loc10: ref %Another = value_as_ref %a.ref
// CHECK:STDOUT:   %addr.loc10: %ptr.289 = addr_of %.loc10
// CHECK:STDOUT:   %H__carbon_thunk.call: init %empty_tuple.type = call imports.%H__carbon_thunk.decl(%addr.loc10)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_explicit_object_param_overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ExplicitObjectParam: type = class_type @ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Another: type = class_type @Another [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.F.cpp_overload_set.type: type = cpp_overload_set_type @ExplicitObjectParam.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.F.cpp_overload_set.value: %ExplicitObjectParam.F.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.7f5: type = ptr_type %ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.1: type = fn_type @F__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.1: %F__carbon_thunk.type.eda1ac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.2: type = fn_type @F__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.2: %F__carbon_thunk.type.eda1ac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.289: type = ptr_type %Another [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.3: type = fn_type @F__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.3: %F__carbon_thunk.type.eda1ac.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ExplicitObjectParam = %ExplicitObjectParam.decl
// CHECK:STDOUT:     .Another = %Another.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitObjectParam.decl: type = class_decl @ExplicitObjectParam [concrete = constants.%ExplicitObjectParam] {} {}
// CHECK:STDOUT:   %Another.decl: type = class_decl @Another [concrete = constants.%Another] {} {}
// CHECK:STDOUT:   %ExplicitObjectParam.F.cpp_overload_set.value: %ExplicitObjectParam.F.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.F.cpp_overload_set [concrete = constants.%ExplicitObjectParam.F.cpp_overload_set.value]
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.1: %F__carbon_thunk.type.eda1ac.1 = fn_decl @F__carbon_thunk.1 [concrete = constants.%F__carbon_thunk.0cd6a8.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.2: %F__carbon_thunk.type.eda1ac.2 = fn_decl @F__carbon_thunk.2 [concrete = constants.%F__carbon_thunk.0cd6a8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.3: %F__carbon_thunk.type.eda1ac.3 = fn_decl @F__carbon_thunk.3 [concrete = constants.%F__carbon_thunk.0cd6a8.3] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%e.param: %ExplicitObjectParam, %n.param: %i32, %a.param: %Another) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %e.ref: %ExplicitObjectParam = name_ref e, %e
// CHECK:STDOUT:   %F.ref.loc8: %ExplicitObjectParam.F.cpp_overload_set.type = name_ref F, imports.%ExplicitObjectParam.F.cpp_overload_set.value [concrete = constants.%ExplicitObjectParam.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %e.ref, %F.ref.loc8
// CHECK:STDOUT:   %.loc8: ref %ExplicitObjectParam = value_as_ref %e.ref
// CHECK:STDOUT:   %addr.loc8: %ptr.7f5 = addr_of %.loc8
// CHECK:STDOUT:   %F__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.e1b8ec.1(%addr.loc8)
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc9: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %F.ref.loc9: %ExplicitObjectParam.F.cpp_overload_set.type = name_ref F, imports.%ExplicitObjectParam.F.cpp_overload_set.value [concrete = constants.%ExplicitObjectParam.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %n.ref, %F.ref.loc9
// CHECK:STDOUT:   %F__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.e1b8ec.2(%n.ref)
// CHECK:STDOUT:   %a.ref: %Another = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc10: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %F.ref.loc10: %ExplicitObjectParam.F.cpp_overload_set.type = name_ref F, imports.%ExplicitObjectParam.F.cpp_overload_set.value [concrete = constants.%ExplicitObjectParam.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref, %F.ref.loc10
// CHECK:STDOUT:   %.loc10: ref %Another = value_as_ref %a.ref
// CHECK:STDOUT:   %addr.loc10: %ptr.289 = addr_of %.loc10
// CHECK:STDOUT:   %F__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.e1b8ec.3(%addr.loc10)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
