// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/method.carbon

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

fn F(v: Cpp.HasQualifiers, p: Cpp.HasQualifiers*) {
  //@dump-sem-ir-begin
  v.const_this();
  v.const_ref_this();

  p->plain();
  p->ref_this();
  p->const_this();
  p->const_ref_this();
  //@dump-sem-ir-end
}

// --- fail_bad_object_param_qualifiers_by_value.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Value(v: Cpp.HasQualifiers) {
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+5]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   v.plain();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  v.plain();

  // TODO: This should remain invalid once we support `volatile`.
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: volatile struct HasQualifiers` [SemanticsTodo]
  // CHECK:STDERR:   v.volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `volatile_this` [InCppNameLookup]
  // CHECK:STDERR:   v.volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  v.volatile_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+5]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   v.ref_this();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  v.ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   v.ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `ref_ref_this` [InCppNameLookup]
  // CHECK:STDERR:   v.ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  v.ref_ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: const struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   v.const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `const_ref_ref_this` [InCppNameLookup]
  // CHECK:STDERR:   v.const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  v.const_ref_ref_this();
}

// --- fail_todo_bad_object_param_qualifiers_by_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Ref(p: Cpp.HasQualifiers*) {
  // TODO: This should eventually be accepted if we support `volatile`.
  // CHECK:STDERR: fail_todo_bad_object_param_qualifiers_by_ref.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: volatile struct HasQualifiers` [SemanticsTodo]
  // CHECK:STDERR:   p->volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_bad_object_param_qualifiers_by_ref.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `volatile_this` [InCppNameLookup]
  // CHECK:STDERR:   p->volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->volatile_this();
}

// --- fail_bad_object_param_qualifiers_ref_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Ref(p: Cpp.HasQualifiers*) {
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   p->ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `ref_ref_this` [InCppNameLookup]
  // CHECK:STDERR:   p->ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->ref_ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: const struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   p->const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `const_ref_ref_this` [InCppNameLookup]
  // CHECK:STDERR:   p->const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->const_ref_ref_this();
}

// CHECK:STDOUT: --- use_object_param_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasQualifiers: type = class_type @HasQualifiers [concrete]
// CHECK:STDOUT:   %ptr.ec3: type = ptr_type %HasQualifiers [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_this.type: type = fn_type @HasQualifiers.const_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_this: %HasQualifiers.const_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.type: type = fn_type @HasQualifiers.const_ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this: %HasQualifiers.const_ref_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.type: type = fn_type @HasQualifiers.plain [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain: %HasQualifiers.plain.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.type: type = fn_type @HasQualifiers.ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this: %HasQualifiers.ref_this.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %HasQualifiers.const_this.decl: %HasQualifiers.const_this.type = fn_decl @HasQualifiers.const_this [concrete = constants.%HasQualifiers.const_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.decl: %HasQualifiers.const_ref_this.type = fn_decl @HasQualifiers.const_ref_this [concrete = constants.%HasQualifiers.const_ref_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.plain.decl: %HasQualifiers.plain.type = fn_decl @HasQualifiers.plain [concrete = constants.%HasQualifiers.plain] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.ref_this.decl: %HasQualifiers.ref_this.type = fn_decl @HasQualifiers.ref_this [concrete = constants.%HasQualifiers.ref_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%v.param: %HasQualifiers, %p.param: %ptr.ec3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref.loc8: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_this.ref.loc8: %HasQualifiers.const_this.type = name_ref const_this, imports.%HasQualifiers.const_this.decl [concrete = constants.%HasQualifiers.const_this]
// CHECK:STDOUT:   %HasQualifiers.const_this.bound.loc8: <bound method> = bound_method %v.ref.loc8, %const_this.ref.loc8
// CHECK:STDOUT:   %HasQualifiers.const_this.call.loc8: init %empty_tuple.type = call %HasQualifiers.const_this.bound.loc8(%v.ref.loc8)
// CHECK:STDOUT:   %v.ref.loc9: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_ref_this.ref.loc9: %HasQualifiers.const_ref_this.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.decl [concrete = constants.%HasQualifiers.const_ref_this]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.bound.loc9: <bound method> = bound_method %v.ref.loc9, %const_ref_this.ref.loc9
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.call.loc9: init %empty_tuple.type = call %HasQualifiers.const_ref_this.bound.loc9(%v.ref.loc9)
// CHECK:STDOUT:   %p.ref.loc11: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc11: ref %HasQualifiers = deref %p.ref.loc11
// CHECK:STDOUT:   %plain.ref: %HasQualifiers.plain.type = name_ref plain, imports.%HasQualifiers.plain.decl [concrete = constants.%HasQualifiers.plain]
// CHECK:STDOUT:   %HasQualifiers.plain.bound: <bound method> = bound_method %.loc11, %plain.ref
// CHECK:STDOUT:   %addr.loc11: %ptr.ec3 = addr_of %.loc11
// CHECK:STDOUT:   %HasQualifiers.plain.call: init %empty_tuple.type = call %HasQualifiers.plain.bound(%addr.loc11)
// CHECK:STDOUT:   %p.ref.loc12: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc12: ref %HasQualifiers = deref %p.ref.loc12
// CHECK:STDOUT:   %ref_this.ref: %HasQualifiers.ref_this.type = name_ref ref_this, imports.%HasQualifiers.ref_this.decl [concrete = constants.%HasQualifiers.ref_this]
// CHECK:STDOUT:   %HasQualifiers.ref_this.bound: <bound method> = bound_method %.loc12, %ref_this.ref
// CHECK:STDOUT:   %addr.loc12: %ptr.ec3 = addr_of %.loc12
// CHECK:STDOUT:   %HasQualifiers.ref_this.call: init %empty_tuple.type = call %HasQualifiers.ref_this.bound(%addr.loc12)
// CHECK:STDOUT:   %p.ref.loc13: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc13_4.1: ref %HasQualifiers = deref %p.ref.loc13
// CHECK:STDOUT:   %const_this.ref.loc13: %HasQualifiers.const_this.type = name_ref const_this, imports.%HasQualifiers.const_this.decl [concrete = constants.%HasQualifiers.const_this]
// CHECK:STDOUT:   %HasQualifiers.const_this.bound.loc13: <bound method> = bound_method %.loc13_4.1, %const_this.ref.loc13
// CHECK:STDOUT:   %.loc13_4.2: %HasQualifiers = bind_value %.loc13_4.1
// CHECK:STDOUT:   %HasQualifiers.const_this.call.loc13: init %empty_tuple.type = call %HasQualifiers.const_this.bound.loc13(%.loc13_4.2)
// CHECK:STDOUT:   %p.ref.loc14: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc14_4.1: ref %HasQualifiers = deref %p.ref.loc14
// CHECK:STDOUT:   %const_ref_this.ref.loc14: %HasQualifiers.const_ref_this.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.decl [concrete = constants.%HasQualifiers.const_ref_this]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.bound.loc14: <bound method> = bound_method %.loc14_4.1, %const_ref_this.ref.loc14
// CHECK:STDOUT:   %.loc14_4.2: %HasQualifiers = bind_value %.loc14_4.1
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.call.loc14: init %empty_tuple.type = call %HasQualifiers.const_ref_this.bound.loc14(%.loc14_4.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
