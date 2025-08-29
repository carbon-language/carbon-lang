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

// --- fail_todo_use_object_param_qualifiers.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn F(v: Cpp.HasQualifiers, p: Cpp.HasQualifiers*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   v.const_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  v.const_this();
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   v.const_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  v.const_ref_this();

  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   p->plain();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  p->plain();
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   p->ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  p->ref_this();
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   p->const_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  p->const_this();
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   p->const_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  p->const_ref_this();
  //@dump-sem-ir-end
}

// --- fail_bad_object_param_qualifiers_by_value.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Value(v: Cpp.HasQualifiers) {
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   v.plain();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  v.plain();

  // TODO: This should remain invalid once we support `volatile`.
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: volatile struct HasQualifiers` [SemanticsTodo]
  // CHECK:STDERR:   v.volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in call to a Cpp function here [InCallToOverloadedCppFunction]
  // CHECK:STDERR:   v.volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  v.volatile_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   v.ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  v.ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   v.ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in call to a Cpp function here [InCallToOverloadedCppFunction]
  // CHECK:STDERR:   v.ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  v.ref_ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: const struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   v.const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in call to a Cpp function here [InCallToOverloadedCppFunction]
  // CHECK:STDERR:   v.const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_bad_object_param_qualifiers_by_ref.carbon:[[@LINE+4]]:3: note: in call to a Cpp function here [InCallToOverloadedCppFunction]
  // CHECK:STDERR:   p->volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->volatile_this();
}

// --- fail_bad_object_param_qualifiers_ref_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Ref(p: Cpp.HasQualifiers*) {
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   p->ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+4]]:3: note: in call to a Cpp function here [InCallToOverloadedCppFunction]
  // CHECK:STDERR:   p->ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->ref_ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: const struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   p->const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+4]]:3: note: in call to a Cpp function here [InCallToOverloadedCppFunction]
  // CHECK:STDERR:   p->const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->const_ref_ref_this();
}

// CHECK:STDOUT: --- fail_todo_use_object_param_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasQualifiers: type = class_type @HasQualifiers [concrete]
// CHECK:STDOUT:   %ptr.ec3: type = ptr_type %HasQualifiers [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.0fd: type = overloaded_cpp_fn_type @HasQualifiers.ref_this [concrete]
// CHECK:STDOUT:   %empty_struct.bce: %.0fd = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_this.type: type = fn_type @HasQualifiers.const_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_this: %HasQualifiers.const_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %.4ae: type = overloaded_cpp_fn_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.930: %.4ae = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.type: type = fn_type @HasQualifiers.const_ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this: %HasQualifiers.const_ref_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b3d: type = overloaded_cpp_fn_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.a36: %.b3d = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.type: type = fn_type @HasQualifiers.plain [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain: %HasQualifiers.plain.type = struct_value () [concrete]
// CHECK:STDOUT:   %.428: type = overloaded_cpp_fn_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.d0d: %.428 = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.type: type = fn_type @HasQualifiers.ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this: %HasQualifiers.ref_this.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %.15f: %.0fd = overloaded_cpp_fn_decl @HasQualifiers.ref_this [concrete = constants.%empty_struct.bce]
// CHECK:STDOUT:   %HasQualifiers.const_this.decl: %HasQualifiers.const_this.type = fn_decl @HasQualifiers.const_this [concrete = constants.%HasQualifiers.const_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.e06: %.4ae = overloaded_cpp_fn_decl @<null name> [concrete = constants.%empty_struct.930]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.decl: %HasQualifiers.const_ref_this.type = fn_decl @HasQualifiers.const_ref_this [concrete = constants.%HasQualifiers.const_ref_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.ed2: %.b3d = overloaded_cpp_fn_decl @<null name> [concrete = constants.%empty_struct.a36]
// CHECK:STDOUT:   %HasQualifiers.plain.decl: %HasQualifiers.plain.type = fn_decl @HasQualifiers.plain [concrete = constants.%HasQualifiers.plain] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.f52: %.428 = overloaded_cpp_fn_decl @<null name> [concrete = constants.%empty_struct.d0d]
// CHECK:STDOUT:   %HasQualifiers.ref_this.decl: %HasQualifiers.ref_this.type = fn_decl @HasQualifiers.ref_this [concrete = constants.%HasQualifiers.ref_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%v.param: %HasQualifiers, %p.param: %ptr.ec3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref.loc13: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_this.ref.loc13: %.0fd = name_ref const_this, imports.%.15f [concrete = constants.%empty_struct.bce]
// CHECK:STDOUT:   %HasQualifiers.const_this.call.loc13: init %empty_tuple.type = call imports.%HasQualifiers.const_this.decl(<error>)
// CHECK:STDOUT:   %v.ref.loc19: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_ref_this.ref.loc19: %.4ae = name_ref const_ref_this, imports.%.e06 [concrete = constants.%empty_struct.930]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.call.loc19: init %empty_tuple.type = call imports.%HasQualifiers.const_ref_this.decl(<error>)
// CHECK:STDOUT:   %p.ref.loc26: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc26: ref %HasQualifiers = deref %p.ref.loc26
// CHECK:STDOUT:   %plain.ref: %.b3d = name_ref plain, imports.%.ed2 [concrete = constants.%empty_struct.a36]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %HasQualifiers.plain.call: init %empty_tuple.type = call imports.%HasQualifiers.plain.decl(<error>)
// CHECK:STDOUT:   %p.ref.loc32: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc32: ref %HasQualifiers = deref %p.ref.loc32
// CHECK:STDOUT:   %ref_this.ref: %.428 = name_ref ref_this, imports.%.f52 [concrete = constants.%empty_struct.d0d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %HasQualifiers.ref_this.call: init %empty_tuple.type = call imports.%HasQualifiers.ref_this.decl(<error>)
// CHECK:STDOUT:   %p.ref.loc38: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc38: ref %HasQualifiers = deref %p.ref.loc38
// CHECK:STDOUT:   %const_this.ref.loc38: %.0fd = name_ref const_this, imports.%.15f [concrete = constants.%empty_struct.bce]
// CHECK:STDOUT:   %HasQualifiers.const_this.call.loc38: init %empty_tuple.type = call imports.%HasQualifiers.const_this.decl(<error>)
// CHECK:STDOUT:   %p.ref.loc44: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc44: ref %HasQualifiers = deref %p.ref.loc44
// CHECK:STDOUT:   %const_ref_this.ref.loc44: %.4ae = name_ref const_ref_this, imports.%.e06 [concrete = constants.%empty_struct.930]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.call.loc44: init %empty_tuple.type = call imports.%HasQualifiers.const_ref_this.decl(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
