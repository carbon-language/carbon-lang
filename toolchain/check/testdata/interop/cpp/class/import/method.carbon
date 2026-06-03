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

import Cpp library "object_param_qualifiers.h";

fn Make() -> Cpp.HasQualifiers;

fn F(v: Cpp.HasQualifiers, p: Cpp.HasQualifiers*) {
  //@dump-sem-ir-begin
  v.const_this();
  v.const_ref_this();
  v.const_ref_ref_this();

  p->plain();
  p->ref_this();
  p->const_this();
  p->const_ref_this();

  Make().plain();
  Make().const_this();
  Make().const_ref_this();
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
  // CHECK:STDERR: fail_todo_bad_object_param_qualifiers_by_ref.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: object parameter type: volatile struct HasQualifiers &` [SemanticsTodo]
  // CHECK:STDERR:   p->volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %ptr.3f7: type = ptr_type %HasQualifiers [concrete]
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
// CHECK:STDOUT:   %const_ref_ref_this__carbon_thunk.type: type = fn_type @const_ref_ref_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %const_ref_ref_this__carbon_thunk: %const_ref_ref_this__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.plain.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.cpp_overload_set.value: %HasQualifiers.plain.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.plain.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.type.9f8225.1: type = fn_type @HasQualifiers.plain.1 [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.64d408.1: %HasQualifiers.plain.type.9f8225.1 = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.cpp_overload_set.value: %HasQualifiers.ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.type: type = fn_type @HasQualifiers.ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this: %HasQualifiers.ref_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.type.9f8225.2: type = fn_type @HasQualifiers.plain.2 [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.64d408.2: %HasQualifiers.plain.type.9f8225.2 = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.cpp_overload_set.type: type = cpp_overload_set_type @HasQualifiers.ref_ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.cpp_overload_set.value: %HasQualifiers.ref_ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.ref_ref_this.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ref_ref_this__carbon_thunk.type: type = fn_type @ref_ref_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %ref_ref_this__carbon_thunk: %ref_ref_this__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %HasQualifiers.HasQualifiers.type: type = fn_type @HasQualifiers.HasQualifiers [concrete]
// CHECK:STDOUT:   %HasQualifiers.HasQualifiers: %HasQualifiers.HasQualifiers.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.0c0: type = const_type %HasQualifiers [concrete]
// CHECK:STDOUT:   %ptr.aa5: type = ptr_type %const.0c0 [concrete]
// CHECK:STDOUT:   %HasQualifiers__carbon_thunk.type: type = fn_type @HasQualifiers__carbon_thunk [concrete]
// CHECK:STDOUT:   %HasQualifiers__carbon_thunk: %HasQualifiers__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.Op.type.43d658.1: type = fn_type @HasQualifiers.Op.1 [concrete]
// CHECK:STDOUT:   %HasQualifiers.Op.cc70d5.1: %HasQualifiers.Op.type.43d658.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b05: <witness> = custom_witness (%HasQualifiers.Op.cc70d5.1), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.0ed: %Copy.type = facet_value %HasQualifiers, (%custom_witness.b05) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.8f4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.0ed) [concrete]
// CHECK:STDOUT:   %.a30: type = fn_type_with_self_type %Copy.WithSelf.Op.type.8f4, %Copy.facet.0ed [concrete]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.type: type = fn_type @HasQualifiers.cpp_destructor [concrete]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor: %HasQualifiers.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.Op.type.43d658.2: type = fn_type @HasQualifiers.Op.2 [concrete]
// CHECK:STDOUT:   %HasQualifiers.Op.cc70d5.2: %HasQualifiers.Op.type.43d658.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %const_ref_ref_this__carbon_thunk.decl: %const_ref_ref_this__carbon_thunk.type = fn_decl @const_ref_ref_this__carbon_thunk [concrete = constants.%const_ref_ref_this__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.plain.cpp_overload_set.value: %HasQualifiers.plain.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.plain.cpp_overload_set [concrete = constants.%HasQualifiers.plain.cpp_overload_set.value]
// CHECK:STDOUT:   %HasQualifiers.plain.decl.3a93fc.1: %HasQualifiers.plain.type.9f8225.1 = fn_decl @HasQualifiers.plain.1 [concrete = constants.%HasQualifiers.plain.64d408.1] {
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
// CHECK:STDOUT:   %HasQualifiers.plain.decl.3a93fc.2: %HasQualifiers.plain.type.9f8225.2 = fn_decl @HasQualifiers.plain.2 [concrete = constants.%HasQualifiers.plain.64d408.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.ref_ref_this.cpp_overload_set.value: %HasQualifiers.ref_ref_this.cpp_overload_set.type = cpp_overload_set_value @HasQualifiers.ref_ref_this.cpp_overload_set [concrete = constants.%HasQualifiers.ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %ref_ref_this__carbon_thunk.decl: %ref_ref_this__carbon_thunk.type = fn_decl @ref_ref_this__carbon_thunk [concrete = constants.%ref_ref_this__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.HasQualifiers.decl: %HasQualifiers.HasQualifiers.type = fn_decl @HasQualifiers.HasQualifiers [concrete = constants.%HasQualifiers.HasQualifiers] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers__carbon_thunk.decl: %HasQualifiers__carbon_thunk.type = fn_decl @HasQualifiers__carbon_thunk [concrete = constants.%HasQualifiers__carbon_thunk] {
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
// CHECK:STDOUT: fn @F(%v.param: %HasQualifiers, %p.param: %ptr.3f7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref.loc10: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_this.ref.loc10: %HasQualifiers.const_this.cpp_overload_set.type = name_ref const_this, imports.%HasQualifiers.const_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %v.ref.loc10, %const_this.ref.loc10
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%v.ref.loc10)
// CHECK:STDOUT:   %v.ref.loc11: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_ref_this.ref.loc11: %HasQualifiers.const_ref_this.cpp_overload_set.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %v.ref.loc11, %const_ref_this.ref.loc11
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%v.ref.loc11)
// CHECK:STDOUT:   %v.ref.loc12: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_ref_ref_this.ref.loc12: %HasQualifiers.const_ref_ref_this.cpp_overload_set.type = name_ref const_ref_ref_this, imports.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %v.ref.loc12, %const_ref_ref_this.ref.loc12
// CHECK:STDOUT:   %const_ref_ref_this__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%const_ref_ref_this__carbon_thunk.decl(%v.ref.loc12)
// CHECK:STDOUT:   %p.ref.loc14: %ptr.3f7 = name_ref p, %p
// CHECK:STDOUT:   %.loc14: ref %HasQualifiers = deref %p.ref.loc14
// CHECK:STDOUT:   %plain.ref.loc14: %HasQualifiers.plain.cpp_overload_set.type = name_ref plain, imports.%HasQualifiers.plain.cpp_overload_set.value [concrete = constants.%HasQualifiers.plain.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %.loc14, %plain.ref.loc14
// CHECK:STDOUT:   %HasQualifiers.plain.call.loc14: init %empty_tuple.type = call imports.%HasQualifiers.plain.decl.3a93fc.1(%.loc14)
// CHECK:STDOUT:   %p.ref.loc15: %ptr.3f7 = name_ref p, %p
// CHECK:STDOUT:   %.loc15: ref %HasQualifiers = deref %p.ref.loc15
// CHECK:STDOUT:   %ref_this.ref: %HasQualifiers.ref_this.cpp_overload_set.type = name_ref ref_this, imports.%HasQualifiers.ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %.loc15, %ref_this.ref
// CHECK:STDOUT:   %HasQualifiers.ref_this.call: init %empty_tuple.type = call imports.%HasQualifiers.ref_this.decl(%.loc15)
// CHECK:STDOUT:   %p.ref.loc16: %ptr.3f7 = name_ref p, %p
// CHECK:STDOUT:   %.loc16_4.1: ref %HasQualifiers = deref %p.ref.loc16
// CHECK:STDOUT:   %const_this.ref.loc16: %HasQualifiers.const_this.cpp_overload_set.type = name_ref const_this, imports.%HasQualifiers.const_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %.loc16_4.1, %const_this.ref.loc16
// CHECK:STDOUT:   %.loc16_4.2: %HasQualifiers = acquire_value %.loc16_4.1
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc16: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%.loc16_4.2)
// CHECK:STDOUT:   %p.ref.loc17: %ptr.3f7 = name_ref p, %p
// CHECK:STDOUT:   %.loc17_4.1: ref %HasQualifiers = deref %p.ref.loc17
// CHECK:STDOUT:   %const_ref_this.ref.loc17: %HasQualifiers.const_ref_this.cpp_overload_set.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc17: <bound method> = bound_method %.loc17_4.1, %const_ref_this.ref.loc17
// CHECK:STDOUT:   %.loc17_4.2: %HasQualifiers = acquire_value %.loc17_4.1
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc17: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%.loc17_4.2)
// CHECK:STDOUT:   %Make.ref.loc19: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc19_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc19: init %HasQualifiers to %.loc19_8.1 = call %Make.ref.loc19()
// CHECK:STDOUT:   %.loc19_8.2: ref %HasQualifiers = temporary %.loc19_8.1, %Make.call.loc19
// CHECK:STDOUT:   %plain.ref.loc19: %HasQualifiers.plain.cpp_overload_set.type = name_ref plain, imports.%HasQualifiers.plain.cpp_overload_set.value [concrete = constants.%HasQualifiers.plain.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %.loc19_8.2, %plain.ref.loc19
// CHECK:STDOUT:   %.loc19_8.3: %HasQualifiers = acquire_value %.loc19_8.2
// CHECK:STDOUT:   %HasQualifiers.plain.call.loc19: init %empty_tuple.type = call imports.%HasQualifiers.plain.decl.3a93fc.2(%.loc19_8.3)
// CHECK:STDOUT:   %Make.ref.loc20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc20_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc20: init %HasQualifiers to %.loc20_8.1 = call %Make.ref.loc20()
// CHECK:STDOUT:   %.loc20_8.2: ref %HasQualifiers = temporary %.loc20_8.1, %Make.call.loc20
// CHECK:STDOUT:   %const_this.ref.loc20: %HasQualifiers.const_this.cpp_overload_set.type = name_ref const_this, imports.%HasQualifiers.const_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc20: <bound method> = bound_method %.loc20_8.2, %const_this.ref.loc20
// CHECK:STDOUT:   %.loc20_8.3: %HasQualifiers = acquire_value %.loc20_8.2
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc20: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%.loc20_8.3)
// CHECK:STDOUT:   %Make.ref.loc21: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc21_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc21: init %HasQualifiers to %.loc21_8.1 = call %Make.ref.loc21()
// CHECK:STDOUT:   %.loc21_8.2: ref %HasQualifiers = temporary %.loc21_8.1, %Make.call.loc21
// CHECK:STDOUT:   %const_ref_this.ref.loc21: %HasQualifiers.const_ref_this.cpp_overload_set.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %.loc21_8.2, %const_ref_this.ref.loc21
// CHECK:STDOUT:   %.loc21_8.3: %HasQualifiers = acquire_value %.loc21_8.2
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc21: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%.loc21_8.3)
// CHECK:STDOUT:   %Make.ref.loc22: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc22_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc22: init %HasQualifiers to %.loc22_8.1 = call %Make.ref.loc22()
// CHECK:STDOUT:   %.loc22_8.2: ref %HasQualifiers = temporary %.loc22_8.1, %Make.call.loc22
// CHECK:STDOUT:   %ref_ref_this.ref: %HasQualifiers.ref_ref_this.cpp_overload_set.type = name_ref ref_ref_this, imports.%HasQualifiers.ref_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc22_9: <bound method> = bound_method %.loc22_8.2, %ref_ref_this.ref
// CHECK:STDOUT:   %.loc22_8.3: %HasQualifiers = acquire_value %.loc22_8.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc22: %.a30 = impl_witness_access constants.%custom_witness.b05, element0 [concrete = constants.%HasQualifiers.Op.cc70d5.1]
// CHECK:STDOUT:   %bound_method.loc22_8: <bound method> = bound_method %.loc22_8.3, %impl.elem0.loc22
// CHECK:STDOUT:   %.loc22_8.4: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Op.ref.loc22_8.1: %HasQualifiers.HasQualifiers.type = name_ref Op, imports.%HasQualifiers.HasQualifiers.decl [concrete = constants.%HasQualifiers.HasQualifiers]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc22_8.5: ref %HasQualifiers = value_as_ref %.loc22_8.3
// CHECK:STDOUT:   %addr.loc22_8.1: %ptr.3f7 = addr_of %.loc22_8.5
// CHECK:STDOUT:   %.loc22_8.6: %ptr.aa5 = as_compatible %addr.loc22_8.1
// CHECK:STDOUT:   %.loc22_8.7: %ptr.aa5 = converted %addr.loc22_8.1, %.loc22_8.6
// CHECK:STDOUT:   %addr.loc22_8.2: %ptr.3f7 = addr_of %self.var
// CHECK:STDOUT:   %HasQualifiers__carbon_thunk.call: init %empty_tuple.type = call imports.%HasQualifiers__carbon_thunk.decl(%.loc22_8.7, %addr.loc22_8.2)
// CHECK:STDOUT:   %.loc22_8.8: init %HasQualifiers to %self.var = mark_in_place_init %HasQualifiers__carbon_thunk.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ref_ref_this__carbon_thunk.call: init %empty_tuple.type = call imports.%ref_ref_this__carbon_thunk.decl(%self.var)
// CHECK:STDOUT:   %Make.ref.loc23: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc23_8.1: ref %HasQualifiers = temporary_storage
// CHECK:STDOUT:   %Make.call.loc23: init %HasQualifiers to %.loc23_8.1 = call %Make.ref.loc23()
// CHECK:STDOUT:   %.loc23_8.2: ref %HasQualifiers = temporary %.loc23_8.1, %Make.call.loc23
// CHECK:STDOUT:   %const_ref_ref_this.ref.loc23: %HasQualifiers.const_ref_ref_this.cpp_overload_set.type = name_ref const_ref_ref_this, imports.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value [concrete = constants.%HasQualifiers.const_ref_ref_this.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %.loc23_8.2, %const_ref_ref_this.ref.loc23
// CHECK:STDOUT:   %.loc23_8.3: %HasQualifiers = acquire_value %.loc23_8.2
// CHECK:STDOUT:   %const_ref_ref_this__carbon_thunk.call.loc23: init %empty_tuple.type = call imports.%const_ref_ref_this__carbon_thunk.decl(%.loc23_8.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc23: <bound method> = bound_method %.loc23_8.2, constants.%HasQualifiers.Op.cc70d5.2
// CHECK:STDOUT:   %Op.ref.loc23: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc23: <bound method> = bound_method %.loc23_8.2, %Op.ref.loc23
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc23: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc23(%.loc23_8.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc22: <bound method> = bound_method %.loc22_8.2, constants.%HasQualifiers.Op.cc70d5.2
// CHECK:STDOUT:   %Op.ref.loc22_8.2: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc22: <bound method> = bound_method %.loc22_8.2, %Op.ref.loc22_8.2
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc22: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc22(%.loc22_8.2)
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc21: <bound method> = bound_method %.loc21_8.2, constants.%HasQualifiers.Op.cc70d5.2
// CHECK:STDOUT:   %Op.ref.loc21: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc21: <bound method> = bound_method %.loc21_8.2, %Op.ref.loc21
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc21: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc21(%.loc21_8.2)
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc20: <bound method> = bound_method %.loc20_8.2, constants.%HasQualifiers.Op.cc70d5.2
// CHECK:STDOUT:   %Op.ref.loc20: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc20: <bound method> = bound_method %.loc20_8.2, %Op.ref.loc20
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc20: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc20(%.loc20_8.2)
// CHECK:STDOUT:   %HasQualifiers.Op.bound.loc19: <bound method> = bound_method %.loc19_8.2, constants.%HasQualifiers.Op.cc70d5.2
// CHECK:STDOUT:   %Op.ref.loc19: %HasQualifiers.cpp_destructor.type = name_ref Op, imports.%HasQualifiers.cpp_destructor.decl [concrete = constants.%HasQualifiers.cpp_destructor]
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.bound.loc19: <bound method> = bound_method %.loc19_8.2, %Op.ref.loc19
// CHECK:STDOUT:   %HasQualifiers.cpp_destructor.call.loc19: init %empty_tuple.type = call %HasQualifiers.cpp_destructor.bound.loc19(%.loc19_8.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_object_param_qualifiers_overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoRefQualifier: type = class_type @NoRefQualifier [concrete]
// CHECK:STDOUT:   %ptr.909: type = ptr_type %NoRefQualifier [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %NoRefQualifier.F.cpp_overload_set.type: type = cpp_overload_set_type @NoRefQualifier.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NoRefQualifier.F.cpp_overload_set.value: %NoRefQualifier.F.cpp_overload_set.type = cpp_overload_set_value @NoRefQualifier.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.715f07.1: type = fn_type @F__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.a50b67.1: %F__carbon_thunk.type.715f07.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.847: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7b5: type = pattern_type %ptr.847 [concrete]
// CHECK:STDOUT:   %NoRefQualifier.F.type.02d694.2: type = fn_type @NoRefQualifier.F.2 [concrete]
// CHECK:STDOUT:   %NoRefQualifier.F.63a4ff.2: %NoRefQualifier.F.type.02d694.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.1: type = fn_type @Destroy.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.1: %Destroy.Op.type.790b3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %WithRefQualifier: type = class_type @WithRefQualifier [concrete]
// CHECK:STDOUT:   %ptr.c4e: type = ptr_type %WithRefQualifier [concrete]
// CHECK:STDOUT:   %WithRefQualifier.F.cpp_overload_set.type: type = cpp_overload_set_type @WithRefQualifier.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %WithRefQualifier.F.cpp_overload_set.value: %WithRefQualifier.F.cpp_overload_set.type = cpp_overload_set_value @WithRefQualifier.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.715f07.2: type = fn_type @F__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.a50b67.2: %F__carbon_thunk.type.715f07.2 = struct_value () [concrete]
// CHECK:STDOUT:   %WithRefQualifier.F.type.02d694.2: type = fn_type @WithRefQualifier.F.2 [concrete]
// CHECK:STDOUT:   %WithRefQualifier.F.63a4ff.2: %WithRefQualifier.F.type.02d694.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %NoRefQualifier.F.cpp_overload_set.value: %NoRefQualifier.F.cpp_overload_set.type = cpp_overload_set_value @NoRefQualifier.F.cpp_overload_set [concrete = constants.%NoRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %F__carbon_thunk.decl.aca65c.1: %F__carbon_thunk.type.715f07.1 = fn_decl @F__carbon_thunk.1 [concrete = constants.%F__carbon_thunk.a50b67.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NoRefQualifier.F.decl.d25af8.2: %NoRefQualifier.F.type.02d694.2 = fn_decl @NoRefQualifier.F.2 [concrete = constants.%NoRefQualifier.F.63a4ff.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithRefQualifier.F.cpp_overload_set.value: %WithRefQualifier.F.cpp_overload_set.type = cpp_overload_set_value @WithRefQualifier.F.cpp_overload_set [concrete = constants.%WithRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %F__carbon_thunk.decl.aca65c.2: %F__carbon_thunk.type.715f07.2 = fn_decl @F__carbon_thunk.2 [concrete = constants.%F__carbon_thunk.a50b67.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithRefQualifier.F.decl.d25af8.2: %WithRefQualifier.F.type.02d694.2 = fn_decl @WithRefQualifier.F.2 [concrete = constants.%WithRefQualifier.F.63a4ff.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFNoRefQualifier(%v.param: %NoRefQualifier, %p.param: %ptr.909) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.104 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.104 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %v.ref: %NoRefQualifier = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc8: %NoRefQualifier.F.cpp_overload_set.type = name_ref F, imports.%NoRefQualifier.F.cpp_overload_set.value [concrete = constants.%NoRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %v.ref, %F.ref.loc8
// CHECK:STDOUT:   %F__carbon_thunk.call: init %i32 = call imports.%F__carbon_thunk.decl.aca65c.1(%v.ref)
// CHECK:STDOUT:   assign %a.var, %F__carbon_thunk.call
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7b5 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7b5 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %ptr.847 = var %b.var_patt
// CHECK:STDOUT:   %p.ref: %ptr.909 = name_ref p, %p
// CHECK:STDOUT:   %.loc9_25: ref %NoRefQualifier = deref %p.ref
// CHECK:STDOUT:   %F.ref.loc9: %NoRefQualifier.F.cpp_overload_set.type = name_ref F, imports.%NoRefQualifier.F.cpp_overload_set.value [concrete = constants.%NoRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_25, %F.ref.loc9
// CHECK:STDOUT:   %NoRefQualifier.F.call: init %ptr.847 = call imports.%NoRefQualifier.F.decl.d25af8.2(%.loc9_25)
// CHECK:STDOUT:   assign %b.var, %NoRefQualifier.F.call
// CHECK:STDOUT:   %.loc9_20: type = splice_block %ptr.loc9 [concrete = constants.%ptr.847] {
// CHECK:STDOUT:     %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %i32.loc9 [concrete = constants.%ptr.847]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ptr.847 = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %b.var, constants.%Destroy.Op.2dd605.1
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9(%self.param: ref %ptr.847) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFWithRefQualifier(%v.param: %WithRefQualifier, %p.param: %ptr.c4e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.104 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.104 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %v.ref: %WithRefQualifier = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc15: %WithRefQualifier.F.cpp_overload_set.type = name_ref F, imports.%WithRefQualifier.F.cpp_overload_set.value [concrete = constants.%WithRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %v.ref, %F.ref.loc15
// CHECK:STDOUT:   %F__carbon_thunk.call: init %i32 = call imports.%F__carbon_thunk.decl.aca65c.2(%v.ref)
// CHECK:STDOUT:   assign %a.var, %F__carbon_thunk.call
// CHECK:STDOUT:   %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7b5 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7b5 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %ptr.847 = var %b.var_patt
// CHECK:STDOUT:   %p.ref: %ptr.c4e = name_ref p, %p
// CHECK:STDOUT:   %.loc16_25: ref %WithRefQualifier = deref %p.ref
// CHECK:STDOUT:   %F.ref.loc16: %WithRefQualifier.F.cpp_overload_set.type = name_ref F, imports.%WithRefQualifier.F.cpp_overload_set.value [concrete = constants.%WithRefQualifier.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %.loc16_25, %F.ref.loc16
// CHECK:STDOUT:   %WithRefQualifier.F.call: init %ptr.847 = call imports.%WithRefQualifier.F.decl.d25af8.2(%.loc16_25)
// CHECK:STDOUT:   assign %b.var, %WithRefQualifier.F.call
// CHECK:STDOUT:   %.loc16_20: type = splice_block %ptr.loc16 [concrete = constants.%ptr.847] {
// CHECK:STDOUT:     %i32.loc16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %i32.loc16 [concrete = constants.%ptr.847]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ptr.847 = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc16: <bound method> = bound_method %b.var, constants.%Destroy.Op.2dd605.1
// CHECK:STDOUT:   %Destroy.Op.call.loc16: init %empty_tuple.type = call %Destroy.Op.bound.loc16(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc15: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.3
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
// CHECK:STDOUT:   %ptr.3e0: type = ptr_type %ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type: type = fn_type @F__carbon_thunk [concrete]
// CHECK:STDOUT:   %F__carbon_thunk: %F__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.G.cpp_overload_set.type: type = cpp_overload_set_type @ExplicitObjectParam.G.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.G.cpp_overload_set.value: %ExplicitObjectParam.G.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.G.cpp_overload_set [concrete]
// CHECK:STDOUT:   %G__carbon_thunk.type: type = fn_type @G__carbon_thunk [concrete]
// CHECK:STDOUT:   %G__carbon_thunk: %G__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.H.cpp_overload_set.type: type = cpp_overload_set_type @ExplicitObjectParam.H.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.H.cpp_overload_set.value: %ExplicitObjectParam.H.cpp_overload_set.type = cpp_overload_set_value @ExplicitObjectParam.H.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.368: type = ptr_type %Another [concrete]
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
// CHECK:STDOUT:   %addr.loc8: %ptr.3e0 = addr_of %.loc8
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
// CHECK:STDOUT:   %addr.loc10: %ptr.368 = addr_of %.loc10
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
// CHECK:STDOUT:   %ptr.3e0: type = ptr_type %ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.715f07.1: type = fn_type @F__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.a50b67.1: %F__carbon_thunk.type.715f07.1 = struct_value () [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.715f07.2: type = fn_type @F__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.a50b67.2: %F__carbon_thunk.type.715f07.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.368: type = ptr_type %Another [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.715f07.3: type = fn_type @F__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.a50b67.3: %F__carbon_thunk.type.715f07.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %F__carbon_thunk.decl.aca65c.1: %F__carbon_thunk.type.715f07.1 = fn_decl @F__carbon_thunk.1 [concrete = constants.%F__carbon_thunk.a50b67.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F__carbon_thunk.decl.aca65c.2: %F__carbon_thunk.type.715f07.2 = fn_decl @F__carbon_thunk.2 [concrete = constants.%F__carbon_thunk.a50b67.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F__carbon_thunk.decl.aca65c.3: %F__carbon_thunk.type.715f07.3 = fn_decl @F__carbon_thunk.3 [concrete = constants.%F__carbon_thunk.a50b67.3] {
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
// CHECK:STDOUT:   %addr.loc8: %ptr.3e0 = addr_of %.loc8
// CHECK:STDOUT:   %F__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.aca65c.1(%addr.loc8)
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc9: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %F.ref.loc9: %ExplicitObjectParam.F.cpp_overload_set.type = name_ref F, imports.%ExplicitObjectParam.F.cpp_overload_set.value [concrete = constants.%ExplicitObjectParam.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %n.ref, %F.ref.loc9
// CHECK:STDOUT:   %F__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.aca65c.2(%n.ref)
// CHECK:STDOUT:   %a.ref: %Another = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc10: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %F.ref.loc10: %ExplicitObjectParam.F.cpp_overload_set.type = name_ref F, imports.%ExplicitObjectParam.F.cpp_overload_set.value [concrete = constants.%ExplicitObjectParam.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.ref, %F.ref.loc10
// CHECK:STDOUT:   %.loc10: ref %Another = value_as_ref %a.ref
// CHECK:STDOUT:   %addr.loc10: %ptr.368 = addr_of %.loc10
// CHECK:STDOUT:   %F__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.aca65c.3(%addr.loc10)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
