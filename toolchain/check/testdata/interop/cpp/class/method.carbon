// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-std=c++23
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

// CHECK:STDOUT: --- use_object_param_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasQualifiers: type = class_type @HasQualifiers [concrete]
// CHECK:STDOUT:   %ptr.ec3: type = ptr_type %HasQualifiers [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_this.type: type = fn_type @HasQualifiers.const_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_this: %HasQualifiers.const_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %HasQualifiers [concrete]
// CHECK:STDOUT:   %ptr.2cb: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %const_this__carbon_thunk.type: type = fn_type @const_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %const_this__carbon_thunk: %const_this__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.type: type = fn_type @HasQualifiers.const_ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this: %HasQualifiers.const_ref_this.type = struct_value () [concrete]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.type: type = fn_type @const_ref_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk: %const_ref_this__carbon_thunk.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %const_this__carbon_thunk.decl: %const_this__carbon_thunk.type = fn_decl @const_this__carbon_thunk [concrete = constants.%const_this__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.decl: %HasQualifiers.const_ref_this.type = fn_decl @HasQualifiers.const_ref_this [concrete = constants.%HasQualifiers.const_ref_this] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.decl: %const_ref_this__carbon_thunk.type = fn_decl @const_ref_this__carbon_thunk [concrete = constants.%const_ref_this__carbon_thunk] {
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
// CHECK:STDOUT:   %.loc8_3: ref %HasQualifiers = value_as_ref %v.ref.loc8
// CHECK:STDOUT:   %addr.loc8: %ptr.ec3 = addr_of %.loc8_3
// CHECK:STDOUT:   %.loc8_16.1: %ptr.2cb = as_compatible %addr.loc8
// CHECK:STDOUT:   %.loc8_16.2: %ptr.2cb = converted %addr.loc8, %.loc8_16.1
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%.loc8_16.2)
// CHECK:STDOUT:   %v.ref.loc9: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_ref_this.ref.loc9: %HasQualifiers.const_ref_this.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.decl [concrete = constants.%HasQualifiers.const_ref_this]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.bound.loc9: <bound method> = bound_method %v.ref.loc9, %const_ref_this.ref.loc9
// CHECK:STDOUT:   %.loc9_3: ref %HasQualifiers = value_as_ref %v.ref.loc9
// CHECK:STDOUT:   %addr.loc9: %ptr.ec3 = addr_of %.loc9_3
// CHECK:STDOUT:   %.loc9_20.1: %ptr.2cb = as_compatible %addr.loc9
// CHECK:STDOUT:   %.loc9_20.2: %ptr.2cb = converted %addr.loc9, %.loc9_20.1
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%.loc9_20.2)
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
// CHECK:STDOUT:   %.loc13_4.3: ref %HasQualifiers = value_as_ref %.loc13_4.2
// CHECK:STDOUT:   %addr.loc13: %ptr.ec3 = addr_of %.loc13_4.3
// CHECK:STDOUT:   %.loc13_17.1: %ptr.2cb = as_compatible %addr.loc13
// CHECK:STDOUT:   %.loc13_17.2: %ptr.2cb = converted %addr.loc13, %.loc13_17.1
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc13: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%.loc13_17.2)
// CHECK:STDOUT:   %p.ref.loc14: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc14_4.1: ref %HasQualifiers = deref %p.ref.loc14
// CHECK:STDOUT:   %const_ref_this.ref.loc14: %HasQualifiers.const_ref_this.type = name_ref const_ref_this, imports.%HasQualifiers.const_ref_this.decl [concrete = constants.%HasQualifiers.const_ref_this]
// CHECK:STDOUT:   %HasQualifiers.const_ref_this.bound.loc14: <bound method> = bound_method %.loc14_4.1, %const_ref_this.ref.loc14
// CHECK:STDOUT:   %.loc14_4.2: %HasQualifiers = bind_value %.loc14_4.1
// CHECK:STDOUT:   %.loc14_4.3: ref %HasQualifiers = value_as_ref %.loc14_4.2
// CHECK:STDOUT:   %addr.loc14: %ptr.ec3 = addr_of %.loc14_4.3
// CHECK:STDOUT:   %.loc14_21.1: %ptr.2cb = as_compatible %addr.loc14
// CHECK:STDOUT:   %.loc14_21.2: %ptr.2cb = converted %addr.loc14, %.loc14_21.1
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc14: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%.loc14_21.2)
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
// CHECK:STDOUT:   %ExplicitObjectParam.F.type: type = fn_type @ExplicitObjectParam.F [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.F: %ExplicitObjectParam.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.7f5: type = ptr_type %ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type: type = fn_type @F__carbon_thunk [concrete]
// CHECK:STDOUT:   %F__carbon_thunk: %F__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.G.type: type = fn_type @ExplicitObjectParam.G [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.G: %ExplicitObjectParam.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.H.type: type = fn_type @ExplicitObjectParam.H [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.H: %ExplicitObjectParam.H.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %ExplicitObjectParam.F.decl: %ExplicitObjectParam.F.type = fn_decl @ExplicitObjectParam.F [concrete = constants.%ExplicitObjectParam.F] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F__carbon_thunk.decl: %F__carbon_thunk.type = fn_decl @F__carbon_thunk [concrete = constants.%F__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitObjectParam.G.decl: %ExplicitObjectParam.G.type = fn_decl @ExplicitObjectParam.G [concrete = constants.%ExplicitObjectParam.G] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitObjectParam.H.decl: %ExplicitObjectParam.H.type = fn_decl @ExplicitObjectParam.H [concrete = constants.%ExplicitObjectParam.H] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %F.ref: %ExplicitObjectParam.F.type = name_ref F, imports.%ExplicitObjectParam.F.decl [concrete = constants.%ExplicitObjectParam.F]
// CHECK:STDOUT:   %ExplicitObjectParam.F.bound: <bound method> = bound_method %e.ref, %F.ref
// CHECK:STDOUT:   %.loc8: ref %ExplicitObjectParam = value_as_ref %e.ref
// CHECK:STDOUT:   %addr.loc8: %ptr.7f5 = addr_of %.loc8
// CHECK:STDOUT:   %F__carbon_thunk.call: init %empty_tuple.type = call imports.%F__carbon_thunk.decl(%addr.loc8)
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc9: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %G.ref: %ExplicitObjectParam.G.type = name_ref G, imports.%ExplicitObjectParam.G.decl [concrete = constants.%ExplicitObjectParam.G]
// CHECK:STDOUT:   %ExplicitObjectParam.G.bound: <bound method> = bound_method %n.ref, %G.ref
// CHECK:STDOUT:   %ExplicitObjectParam.G.call: init %empty_tuple.type = call %ExplicitObjectParam.G.bound(%n.ref)
// CHECK:STDOUT:   %a.ref: %Another = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc10: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %H.ref: %ExplicitObjectParam.H.type = name_ref H, imports.%ExplicitObjectParam.H.decl [concrete = constants.%ExplicitObjectParam.H]
// CHECK:STDOUT:   %ExplicitObjectParam.H.bound: <bound method> = bound_method %a.ref, %H.ref
// CHECK:STDOUT:   %.loc10: ref %Another = value_as_ref %a.ref
// CHECK:STDOUT:   %addr.loc10: %ptr.289 = addr_of %.loc10
// CHECK:STDOUT:   %H__carbon_thunk.call: init %empty_tuple.type = call imports.%H__carbon_thunk.decl(%addr.loc10)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
