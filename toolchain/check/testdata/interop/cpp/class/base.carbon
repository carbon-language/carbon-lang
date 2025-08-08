// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/base.carbon

// --- derived_to_base_conversion.h

class Base {};
class Derived : public Base {};

// --- use_derived_to_base_conversion.carbon

library "[[@TEST_NAME]]";

import Cpp library "derived_to_base_conversion.h";

//@dump-sem-ir-begin
fn ConvertPtr(d: Cpp.Derived*) -> Cpp.Base* {
  return d;
}
//@dump-sem-ir-end

fn AcceptVal(b: Cpp.Base);

//@dump-sem-ir-begin
fn ConvertVal(d: Cpp.Derived) {
  AcceptVal(d);
}
//@dump-sem-ir-end

// --- static_member.h

class Base {
 public:
  static auto base_fn() -> void;
};

class Derived : public Base {
 public:
  static auto derived_fn() -> void;
};

// --- use_static_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "static_member.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.Base.base_fn();
  Cpp.Derived.base_fn();
  Cpp.Derived.derived_fn();
  //@dump-sem-ir-end
}

// --- base_field.h

struct Base {
  int a;
  int b;
};

struct Derived : Base {
  int b;
};

// --- use_base_field.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_field.h";

fn AccessDirect(d: Cpp.Derived) -> i32 {
  //@dump-sem-ir-begin
  return d.a;
  //@dump-sem-ir-end
}

fn AccessQualified(d: Cpp.Derived) -> i32 {
  //@dump-sem-ir-begin
  return d.(Cpp.Base.b);
  //@dump-sem-ir-end
}

// --- base_method.h

struct Base {
  void f() const;
  void g() const;
};

struct Derived : Base {
  void g() const;
};

// --- fail_todo_use_base_method.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_method.h";

fn CallDirect(d: Cpp.Derived) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_base_method.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   d.f();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_todo_use_base_method.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  d.f();
  //@dump-sem-ir-end
}

fn CallQualified(d: Cpp.Derived) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_base_method.carbon:[[@LINE+9]]:3: error: member name of type `<type of Cpp.g>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   d.(Cpp.Base.g)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_base_method.carbon:[[@LINE+5]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   d.(Cpp.Base.g)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_base_method.carbon: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:
  d.(Cpp.Base.g)();
  //@dump-sem-ir-end
}

// --- multiple_inheritance.h

struct A { int a; };
struct B { int b; };
struct C : A, B {};

// --- fail_todo_use_multiple_inheritance.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_inheritance.h";

fn ConvertA(p: Cpp.C*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.C*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.C*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

fn ConvertB(p: Cpp.C*) -> Cpp.B* {
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.C*` to `Cpp.B*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_multiple_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.C*` does not implement interface `Core.ImplicitAs(Cpp.B*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- virtual_inheritance.h

struct A { int a; };
struct B : virtual A {};

void UseB(B * _Nonnull p);

// --- use_virtual_inheritance_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "virtual_inheritance.h";

fn Convert(p: Cpp.B*) {
  // OK, doesn't require `Cpp.B` to be a complete type.
  Cpp.UseB(p);
}

// --- fail_todo_use_virtual_inheritance.carbon

library "[[@TEST_NAME]]";

import Cpp library "virtual_inheritance.h";

fn Convert(p: Cpp.B*) -> Cpp.A* {
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+14]]:3: error: semantics TODO: `class with virtual bases` [SemanticsTodo]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+11]]:3: note: while completing C++ class type `Cpp.B` [InCppTypeCompletion]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Cpp.B*` to `Cpp.A*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_use_virtual_inheritance.carbon:[[@LINE+4]]:3: note: type `Cpp.B*` does not implement interface `Core.ImplicitAs(Cpp.A*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- final.h

struct A final {};

// --- fail_derive_from_final.carbon

library "[[@TEST_NAME]]";

import Cpp library "final.h";

class B {
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+4]]:16: error: deriving from final type `A`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: Cpp.A;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: Cpp.A;
}

// --- union.h

union U {};

// --- fail_derive_from_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "union.h";

class V {
  // CHECK:STDERR: fail_derive_from_union.carbon:[[@LINE+4]]:16: error: deriving from final type `U`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: Cpp.U;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: Cpp.U;
}

// CHECK:STDOUT: --- use_derived_to_base_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %ptr.ddb: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.5c8: type = pattern_type %ptr.ddb [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.fb2: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.72a: type = pattern_type %ptr.fb2 [concrete]
// CHECK:STDOUT:   %ConvertPtr.type: type = fn_type @ConvertPtr [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ConvertPtr: %ConvertPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %AcceptVal.type: type = fn_type @AcceptVal [concrete]
// CHECK:STDOUT:   %AcceptVal: %AcceptVal.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.5de: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %ConvertVal.type: type = fn_type @ConvertVal [concrete]
// CHECK:STDOUT:   %ConvertVal: %ConvertVal.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %ConvertPtr.decl: %ConvertPtr.type = fn_decl @ConvertPtr [concrete = constants.%ConvertPtr] {
// CHECK:STDOUT:     %d.patt: %pattern_type.5c8 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.5c8 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.72a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.72a = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Cpp.ref.loc7_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %ptr.loc7_43: type = ptr_type %Base.ref [concrete = constants.%ptr.fb2]
// CHECK:STDOUT:     %d.param: %ptr.ddb = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr.loc7_29 [concrete = constants.%ptr.ddb] {
// CHECK:STDOUT:       %Cpp.ref.loc7_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc7_29: type = ptr_type %Derived.ref [concrete = constants.%ptr.ddb]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %ptr.ddb = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %ptr.fb2 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.fb2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertVal.decl: %ConvertVal.type = fn_decl @ConvertVal [concrete = constants.%ConvertVal] {
// CHECK:STDOUT:     %d.patt: %pattern_type.5de = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.5de = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %Derived.ref [concrete = constants.%Derived] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertPtr(%d.param: %ptr.ddb) -> %ptr.fb2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %ptr.ddb = name_ref d, %d
// CHECK:STDOUT:   %.loc8_11.1: ref %Derived = deref %d.ref
// CHECK:STDOUT:   %.loc8_11.2: ref %Base = class_element_access %.loc8_11.1, element0
// CHECK:STDOUT:   %addr: %ptr.fb2 = addr_of %.loc8_11.2
// CHECK:STDOUT:   %.loc8_11.3: %ptr.fb2 = converted %d.ref, %addr
// CHECK:STDOUT:   return %.loc8_11.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertVal(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AcceptVal.ref: %AcceptVal.type = name_ref AcceptVal, file.%AcceptVal.decl [concrete = constants.%AcceptVal]
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %.loc16_13.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc16_13.2: ref %Base = converted %d.ref, %.loc16_13.1
// CHECK:STDOUT:   %.loc16_13.3: %Base = bind_value %.loc16_13.2
// CHECK:STDOUT:   %AcceptVal.call: init %empty_tuple.type = call %AcceptVal.ref(%.loc16_13.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_static_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %.fc7: type = overloaded_cpp_fn_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.4df: %.fc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Base.base_fn.type: type = fn_type @Base.base_fn [concrete]
// CHECK:STDOUT:   %Base.base_fn: %Base.base_fn.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %.243: type = overloaded_cpp_fn_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.e3a: %.243 = struct_value () [concrete]
// CHECK:STDOUT:   %.81e: type = overloaded_cpp_fn_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.5c0: %.81e = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.derived_fn.type: type = fn_type @Derived.derived_fn [concrete]
// CHECK:STDOUT:   %Derived.derived_fn: %Derived.derived_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %.511: %.fc7 = overloaded_cpp_fn_decl @<null name> [concrete = constants.%empty_struct.4df]
// CHECK:STDOUT:   %Base.base_fn.decl: %Base.base_fn.type = fn_decl @Base.base_fn [concrete = constants.%Base.base_fn] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %.12f: %.243 = overloaded_cpp_fn_decl @<null name> [concrete = constants.%empty_struct.e3a]
// CHECK:STDOUT:   %.064: %.81e = overloaded_cpp_fn_decl @<null name> [concrete = constants.%empty_struct.5c0]
// CHECK:STDOUT:   %Derived.derived_fn.decl: %Derived.derived_fn.type = fn_decl @Derived.derived_fn [concrete = constants.%Derived.derived_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %base_fn.ref.loc8: %.fc7 = name_ref base_fn, imports.%.511 [concrete = constants.%empty_struct.4df]
// CHECK:STDOUT:   %Base.base_fn.call.loc8: init %empty_tuple.type = call imports.%Base.base_fn.decl()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Derived.ref.loc9: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %base_fn.ref.loc9: %.243 = name_ref base_fn, imports.%.12f [concrete = constants.%empty_struct.e3a]
// CHECK:STDOUT:   %Base.base_fn.call.loc9: init %empty_tuple.type = call imports.%Base.base_fn.decl()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Derived.ref.loc10: type = name_ref Derived, imports.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %derived_fn.ref: %.81e = name_ref derived_fn, imports.%.064 [concrete = constants.%empty_struct.5c0]
// CHECK:STDOUT:   %Derived.derived_fn.call: init %empty_tuple.type = call imports.%Derived.derived_fn.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDirect(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.1 [concrete = @Base.%.1]
// CHECK:STDOUT:   %.loc8_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: ref %Base = converted %d.ref, %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.3: ref %i32 = class_element_access %.loc8_11.2, element0
// CHECK:STDOUT:   %.loc8_11.4: %i32 = bind_value %.loc8_11.3
// CHECK:STDOUT:   return %.loc8_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessQualified(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.2 [concrete = @Base.%.2]
// CHECK:STDOUT:   %.loc14_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: ref %Base = converted %d.ref, %.loc14_11.1
// CHECK:STDOUT:   %.loc14_11.3: ref %i32 = class_element_access %.loc14_11.2, element1
// CHECK:STDOUT:   %.loc14_11.4: %i32 = bind_value %.loc14_11.3
// CHECK:STDOUT:   return %.loc14_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %.3cf: type = overloaded_cpp_fn_type @CallQualified [concrete]
// CHECK:STDOUT:   %empty_struct.d07: %.3cf = struct_value () [concrete]
// CHECK:STDOUT:   %Base.f.type: type = fn_type @Base.f [concrete]
// CHECK:STDOUT:   %Base.f: %Base.f.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b92: type = overloaded_cpp_fn_type @Base.g [concrete]
// CHECK:STDOUT:   %empty_struct.271: %.b92 = struct_value () [concrete]
// CHECK:STDOUT:   %Base.g.type: type = fn_type @Base.g [concrete]
// CHECK:STDOUT:   %Base.g: %Base.g.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %.c0b: %.3cf = overloaded_cpp_fn_decl @CallQualified [concrete = constants.%empty_struct.d07]
// CHECK:STDOUT:   %Base.f.decl: %Base.f.type = fn_decl @Base.f [concrete = constants.%Base.f] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7b4: %.b92 = overloaded_cpp_fn_decl @Base.g [concrete = constants.%empty_struct.271]
// CHECK:STDOUT:   %Base.g.decl: %Base.g.type = fn_decl @Base.g [concrete = constants.%Base.g] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallDirect(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %f.ref: %.3cf = name_ref f, imports.%.c0b [concrete = constants.%empty_struct.d07]
// CHECK:STDOUT:   %Base.f.call: init %empty_tuple.type = call imports.%Base.f.decl(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallQualified(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %g.ref: %.b92 = name_ref g, imports.%.7b4 [concrete = constants.%empty_struct.271]
// CHECK:STDOUT:   %Base.g.call: init %empty_tuple.type = call imports.%Base.g.decl(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
