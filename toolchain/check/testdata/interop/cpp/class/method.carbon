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

// TODO: Tests marked as `fail_todo_5891_` to fixed as a follow-up of https://github.com/carbon-language/carbon-lang/pull/5891.

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

// --- fail_todo_5891_use_object_param_qualifiers.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn F(v: Cpp.HasQualifiers, p: Cpp.HasQualifiers*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   v.const_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  v.const_this();
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   v.const_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  v.const_ref_this();

  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   p->plain();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  p->plain();
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   p->ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  p->ref_this();
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   p->const_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  p->const_this();
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   p->const_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_use_object_param_qualifiers.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  p->const_ref_this();
  //@dump-sem-ir-end
}

// --- fail_bad_5891_object_param_qualifiers_by_value.carbon

library "[[@TEST_NAME]]";

import Cpp library "object_param_qualifiers.h";

fn Value(v: Cpp.HasQualifiers) {
  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   v.plain();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  v.plain();

  // TODO: This should remain invalid once we support `volatile`.
  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: volatile struct HasQualifiers` [SemanticsTodo]
  // CHECK:STDERR:   v.volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   v.volatile_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  v.volatile_this();

  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   v.ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  v.ref_this();

  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   v.ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   v.ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  v.ref_ref_this();

  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: const struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   v.const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_5891_object_param_qualifiers_by_value.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
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
  // CHECK:STDERR: fail_todo_bad_object_param_qualifiers_by_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
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
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   p->ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->ref_ref_this();

  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: object parameter type: const struct HasQualifiers &&` [SemanticsTodo]
  // CHECK:STDERR:   p->const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_object_param_qualifiers_ref_ref.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   p->const_ref_ref_this();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
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

// --- fail_todo_5891_call_explicit_object_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "explicit_object_param.h";

fn Call(e: Cpp.ExplicitObjectParam, n: i32, a: Cpp.Another) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_5891_call_explicit_object_param.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   e.F();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_todo_5891_call_explicit_object_param.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  e.F();
  // CHECK:STDERR: fail_todo_5891_call_explicit_object_param.carbon:[[@LINE+9]]:3: error: member name of type `<type of Cpp.G>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   n.(Cpp.ExplicitObjectParam.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_5891_call_explicit_object_param.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   n.(Cpp.ExplicitObjectParam.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_call_explicit_object_param.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  n.(Cpp.ExplicitObjectParam.G)();
  // CHECK:STDERR: fail_todo_5891_call_explicit_object_param.carbon:[[@LINE+9]]:3: error: member name of type `<type of Cpp.H>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   a.(Cpp.ExplicitObjectParam.H)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_5891_call_explicit_object_param.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   a.(Cpp.ExplicitObjectParam.H)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_call_explicit_object_param.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  a.(Cpp.ExplicitObjectParam.H)();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_5891_use_object_param_qualifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasQualifiers: type = class_type @HasQualifiers [concrete]
// CHECK:STDOUT:   %ptr.ec3: type = ptr_type %HasQualifiers [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.602: type = cpp_overload_set_type @const_ref_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %empty_struct.63b: %.602 = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %HasQualifiers [concrete]
// CHECK:STDOUT:   %ptr.2cb: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %const_this__carbon_thunk.type: type = fn_type @const_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %const_this__carbon_thunk: %const_this__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.bf0: type = cpp_overload_set_type @HasQualifiers.plain [concrete]
// CHECK:STDOUT:   %empty_struct.4f2: %.bf0 = struct_value () [concrete]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.type: type = fn_type @const_ref_this__carbon_thunk [concrete]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk: %const_ref_this__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.64b: type = cpp_overload_set_type @HasQualifiers.ref_this [concrete]
// CHECK:STDOUT:   %empty_struct.a83: %.64b = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain.type: type = fn_type @HasQualifiers.plain [concrete]
// CHECK:STDOUT:   %HasQualifiers.plain: %HasQualifiers.plain.type = struct_value () [concrete]
// CHECK:STDOUT:   %.352: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.f58: %.352 = struct_value () [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this.type: type = fn_type @HasQualifiers.ref_this [concrete]
// CHECK:STDOUT:   %HasQualifiers.ref_this: %HasQualifiers.ref_this.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %.ce8: %.602 = cpp_overload_set_value @const_ref_this__carbon_thunk [concrete = constants.%empty_struct.63b]
// CHECK:STDOUT:   %const_this__carbon_thunk.decl: %const_this__carbon_thunk.type = fn_decl @const_this__carbon_thunk [concrete = constants.%const_this__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.667: %.bf0 = cpp_overload_set_value @HasQualifiers.plain [concrete = constants.%empty_struct.4f2]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.decl: %const_ref_this__carbon_thunk.type = fn_decl @const_ref_this__carbon_thunk [concrete = constants.%const_ref_this__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.646: %.64b = cpp_overload_set_value @HasQualifiers.ref_this [concrete = constants.%empty_struct.a83]
// CHECK:STDOUT:   %HasQualifiers.plain.decl: %HasQualifiers.plain.type = fn_decl @HasQualifiers.plain [concrete = constants.%HasQualifiers.plain] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.145: %.352 = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct.f58]
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
// CHECK:STDOUT:   %const_this.ref.loc13: %.602 = name_ref const_this, imports.%.ce8 [concrete = constants.%empty_struct.63b]
// CHECK:STDOUT:   %addr.loc13: %ptr.ec3 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc13_16.1: %ptr.2cb = as_compatible %addr.loc13 [concrete = <error>]
// CHECK:STDOUT:   %.loc13_16.2: %ptr.2cb = converted %addr.loc13, %.loc13_16.1 [concrete = <error>]
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc13: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%.loc13_16.2)
// CHECK:STDOUT:   %v.ref.loc19: %HasQualifiers = name_ref v, %v
// CHECK:STDOUT:   %const_ref_this.ref.loc19: %.bf0 = name_ref const_ref_this, imports.%.667 [concrete = constants.%empty_struct.4f2]
// CHECK:STDOUT:   %addr.loc19: %ptr.ec3 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc19_20.1: %ptr.2cb = as_compatible %addr.loc19 [concrete = <error>]
// CHECK:STDOUT:   %.loc19_20.2: %ptr.2cb = converted %addr.loc19, %.loc19_20.1 [concrete = <error>]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc19: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%.loc19_20.2)
// CHECK:STDOUT:   %p.ref.loc26: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc26: ref %HasQualifiers = deref %p.ref.loc26
// CHECK:STDOUT:   %plain.ref: %.64b = name_ref plain, imports.%.646 [concrete = constants.%empty_struct.a83]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %HasQualifiers.plain.call: init %empty_tuple.type = call imports.%HasQualifiers.plain.decl(<error>)
// CHECK:STDOUT:   %p.ref.loc32: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc32: ref %HasQualifiers = deref %p.ref.loc32
// CHECK:STDOUT:   %ref_this.ref: %.352 = name_ref ref_this, imports.%.145 [concrete = constants.%empty_struct.f58]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %HasQualifiers.ref_this.call: init %empty_tuple.type = call imports.%HasQualifiers.ref_this.decl(<error>)
// CHECK:STDOUT:   %p.ref.loc38: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc38_4: ref %HasQualifiers = deref %p.ref.loc38
// CHECK:STDOUT:   %const_this.ref.loc38: %.602 = name_ref const_this, imports.%.ce8 [concrete = constants.%empty_struct.63b]
// CHECK:STDOUT:   %addr.loc38: %ptr.ec3 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc38_17.1: %ptr.2cb = as_compatible %addr.loc38 [concrete = <error>]
// CHECK:STDOUT:   %.loc38_17.2: %ptr.2cb = converted %addr.loc38, %.loc38_17.1 [concrete = <error>]
// CHECK:STDOUT:   %const_this__carbon_thunk.call.loc38: init %empty_tuple.type = call imports.%const_this__carbon_thunk.decl(%.loc38_17.2)
// CHECK:STDOUT:   %p.ref.loc44: %ptr.ec3 = name_ref p, %p
// CHECK:STDOUT:   %.loc44_4: ref %HasQualifiers = deref %p.ref.loc44
// CHECK:STDOUT:   %const_ref_this.ref.loc44: %.bf0 = name_ref const_ref_this, imports.%.667 [concrete = constants.%empty_struct.4f2]
// CHECK:STDOUT:   %addr.loc44: %ptr.ec3 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc44_21.1: %ptr.2cb = as_compatible %addr.loc44 [concrete = <error>]
// CHECK:STDOUT:   %.loc44_21.2: %ptr.2cb = converted %addr.loc44, %.loc44_21.1 [concrete = <error>]
// CHECK:STDOUT:   %const_ref_this__carbon_thunk.call.loc44: init %empty_tuple.type = call imports.%const_ref_this__carbon_thunk.decl(%.loc44_21.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_5891_call_explicit_object_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ExplicitObjectParam: type = class_type @ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Another: type = class_type @Another [concrete]
// CHECK:STDOUT:   %.3e2: type = cpp_overload_set_type @ExplicitObjectParam.G [concrete]
// CHECK:STDOUT:   %empty_struct.48d: %.3e2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.7f5: type = ptr_type %ExplicitObjectParam [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type: type = fn_type @F__carbon_thunk [concrete]
// CHECK:STDOUT:   %F__carbon_thunk: %F__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.d44: type = cpp_overload_set_type @ExplicitObjectParam.H [concrete]
// CHECK:STDOUT:   %empty_struct.a7c: %.d44 = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.G.type: type = fn_type @ExplicitObjectParam.G [concrete]
// CHECK:STDOUT:   %ExplicitObjectParam.G: %ExplicitObjectParam.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.9fb: type = cpp_overload_set_type @H__carbon_thunk [concrete]
// CHECK:STDOUT:   %empty_struct.b14: %.9fb = struct_value () [concrete]
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
// CHECK:STDOUT:   %.037: %.3e2 = cpp_overload_set_value @ExplicitObjectParam.G [concrete = constants.%empty_struct.48d]
// CHECK:STDOUT:   %F__carbon_thunk.decl: %F__carbon_thunk.type = fn_decl @F__carbon_thunk [concrete = constants.%F__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.151: %.d44 = cpp_overload_set_value @ExplicitObjectParam.H [concrete = constants.%empty_struct.a7c]
// CHECK:STDOUT:   %ExplicitObjectParam.G.decl: %ExplicitObjectParam.G.type = fn_decl @ExplicitObjectParam.G [concrete = constants.%ExplicitObjectParam.G] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.34a: %.9fb = cpp_overload_set_value @H__carbon_thunk [concrete = constants.%empty_struct.b14]
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
// CHECK:STDOUT:   %F.ref: %.3e2 = name_ref F, imports.%.037 [concrete = constants.%empty_struct.48d]
// CHECK:STDOUT:   %addr.loc13: %ptr.7f5 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %F__carbon_thunk.call: init %empty_tuple.type = call imports.%F__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc23: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %G.ref: %.d44 = name_ref G, imports.%.151 [concrete = constants.%empty_struct.a7c]
// CHECK:STDOUT:   %ExplicitObjectParam.G.call: init %empty_tuple.type = call imports.%ExplicitObjectParam.G.decl(<error>)
// CHECK:STDOUT:   %a.ref: %Another = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref.loc33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ExplicitObjectParam.ref.loc33: type = name_ref ExplicitObjectParam, imports.%ExplicitObjectParam.decl [concrete = constants.%ExplicitObjectParam]
// CHECK:STDOUT:   %H.ref: %.9fb = name_ref H, imports.%.34a [concrete = constants.%empty_struct.b14]
// CHECK:STDOUT:   %addr.loc33: %ptr.289 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %H__carbon_thunk.call: init %empty_tuple.type = call imports.%H__carbon_thunk.decl(%addr.loc33)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
