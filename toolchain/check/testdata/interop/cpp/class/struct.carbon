// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/struct.carbon

// ============================================================================
// Declaration
// ============================================================================

// --- declaration.h

struct Bar;

// --- fail_todo_import_declaration.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration.h";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_import_declaration.carbon:[[@LINE+7]]:13: error: semantics TODO: `Unsupported: Record declarations without a definition` [SemanticsTodo]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_todo_import_declaration.carbon:[[@LINE+4]]:13: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn MyF(bar: Cpp.Bar*);
//@dump-sem-ir-end

// ============================================================================
// Definition
// ============================================================================

// --- definition.h

struct Bar {};

// --- import_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.Bar*);
//@dump-sem-ir-end

// ============================================================================
// Declaration and definition
// ============================================================================

// --- declaration_and_definition.h

struct Bar;
struct Bar {};

// --- import_declaration_and_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration_and_definition.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.Bar*);
//@dump-sem-ir-end

// ============================================================================
// Public static member function
// ============================================================================

// --- public_static_member_function.h

struct Bar {
  static auto foo() -> void;
};

// --- import_public_static_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "public_static_member_function.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.Bar.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Private static member function
// ============================================================================

// --- private_static_member_function.h

struct Bar {
 private:
  static auto foo() -> void;
};

// --- todo_fail_import_private_static_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "private_static_member_function.h";

fn MyF() {
  Cpp.Bar.foo();
}

// ============================================================================
// Public static data member
// ============================================================================

// --- public_static_data_member.h

struct Bar {
  static Bar* foo;
};

// --- fail_todo_import_public_static_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "public_static_data_member.h";

fn MyF() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_public_static_data_member.carbon:[[@LINE+11]]:23: error: semantics TODO: `Unsupported: Declaration type Var` [SemanticsTodo]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_public_static_data_member.carbon:[[@LINE+8]]:23: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_public_static_data_member.carbon:[[@LINE+4]]:23: error: member name `foo` not found in `Cpp.Bar` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let bar: Cpp.Bar* = Cpp.Bar.foo();
  // CHECK:STDERR:                       ^~~~~~~~~~~
  // CHECK:STDERR:
  let bar: Cpp.Bar* = Cpp.Bar.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Public data member
// ============================================================================

// --- public_data_member.h

struct Bar {
  Bar* foo;
};

// --- fail_todo_import_public_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "public_data_member.h";

//@dump-sem-ir-begin
fn MyF(bar : Cpp.Bar*) {
  // CHECK:STDERR: fail_todo_import_public_data_member.carbon:[[@LINE+11]]:27: error: semantics TODO: `Unsupported: Declaration type Field` [SemanticsTodo]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR: fail_todo_import_public_data_member.carbon:[[@LINE+8]]:27: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_public_data_member.carbon:[[@LINE+4]]:27: error: member name `foo` not found in `Cpp.Bar` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let foo_bar: Cpp.Bar* = bar->foo;
  // CHECK:STDERR:                           ^~~~~~~~
  // CHECK:STDERR:
  let foo_bar: Cpp.Bar* = bar->foo;
}
//@dump-sem-ir-end

// ============================================================================
// Inheritance static
// ============================================================================

// --- inheritance_static.h

struct Bar1 {
  static auto foo1() -> void;
};

struct Bar2 : public Bar1 {
  static auto foo2() -> void;
};

// --- import_inheritance_static.carbon

library "[[@TEST_NAME]]";

import Cpp library "inheritance_static.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.Bar1.foo1();
  Cpp.Bar2.foo1();
  Cpp.Bar2.foo2();
  //@dump-sem-ir-end
}

// ============================================================================
// Inheritance pointers
// ============================================================================

// --- inheritance_pointers.h

struct Bar1 {};
struct Bar2 : public Bar1 {};

// --- fail_todo_import_inheritance_pointers.carbon

library "[[@TEST_NAME]]";

import Cpp library "inheritance_pointers.h";

//@dump-sem-ir-begin
fn MyF1(bar : Cpp.Bar1*);
// TODO: Support C++ inheritance.
// CHECK:STDERR: fail_todo_import_inheritance_pointers.carbon:[[@LINE+10]]:33: error: cannot implicitly convert expression of type `Cpp.Bar2*` to `Cpp.Bar1*` [ConversionFailure]
// CHECK:STDERR: fn MyF2(bar : Cpp.Bar2*) { MyF1(bar); }
// CHECK:STDERR:                                 ^~~
// CHECK:STDERR: fail_todo_import_inheritance_pointers.carbon:[[@LINE+7]]:33: note: type `Cpp.Bar2*` does not implement interface `Core.ImplicitAs(Cpp.Bar1*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn MyF2(bar : Cpp.Bar2*) { MyF1(bar); }
// CHECK:STDERR:                                 ^~~
// CHECK:STDERR: fail_todo_import_inheritance_pointers.carbon:[[@LINE-8]]:9: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn MyF1(bar : Cpp.Bar1*);
// CHECK:STDERR:         ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MyF2(bar : Cpp.Bar2*) { MyF1(bar); }
//@dump-sem-ir-end

// ============================================================================
// Dynamic
// ============================================================================

// --- dynamic.h

struct Bar {
  virtual ~Bar();
};

// --- fail_todo_import_dynamic.carbon

library "[[@TEST_NAME]]";

import Cpp library "dynamic.h";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_import_dynamic.carbon:[[@LINE+7]]:14: error: semantics TODO: `Unsupported: Dynamic Class` [SemanticsTodo]
// CHECK:STDERR: fn MyF(bar : Cpp.Bar*);
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR: fail_todo_import_dynamic.carbon:[[@LINE+4]]:14: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar : Cpp.Bar*);
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
fn MyF(bar : Cpp.Bar*);
//@dump-sem-ir-end

// ============================================================================
// To inherit public
// ============================================================================

// --- to_inherit_public.h

struct Bar { static auto foo() -> void; };

// --- import_to_inherit_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "to_inherit_public.h";

//@dump-sem-ir-begin
class Derived {
  extend base: Cpp.Bar;
}

fn MyF() {
  Derived.foo();
}
//@dump-sem-ir-end

// ============================================================================
// To inherit private
// ============================================================================

// --- to_inherit_private.h

struct Bar {
 private:
  static auto foo() -> void;
};

// --- todo_fail_to_inherit_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "to_inherit_private.h";

class Derived {
  extend base: Cpp.Bar;
}

fn MyF() {
  Derived.foo();
}

// ============================================================================
// Template
// ============================================================================

// --- template.h

template<typename T>
struct Bar {};

// --- fail_todo_import_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "template.h";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_import_template.carbon:[[@LINE+11]]:13: error: semantics TODO: `Unsupported: Declaration type ClassTemplate` [SemanticsTodo]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_todo_import_template.carbon:[[@LINE+8]]:13: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_import_template.carbon:[[@LINE+4]]:13: error: member name `Bar` not found in `Cpp` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn MyF(bar: Cpp.Bar*);
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_todo_import_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: <error> = binding_pattern bar [concrete]
// CHECK:STDOUT:     %bar.param_patt: <error> = value_param_pattern %bar.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc14: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.ref: <error> = name_ref Bar, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: <error> = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: %pattern_type = binding_pattern bar [concrete]
// CHECK:STDOUT:     %bar.param_patt: %pattern_type = value_param_pattern %bar.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_declaration_and_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: %pattern_type = binding_pattern bar [concrete]
// CHECK:STDOUT:     %bar.param_patt: %pattern_type = value_param_pattern %bar.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_public_static_member_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_public_static_data_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %ptr.f68: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.f68 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bar.patt: %pattern_type = binding_pattern bar [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar.ref.loc19_26: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc19: type = splice_block %ptr [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:     %Cpp.ref.loc19_12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Bar.ref.loc19_15: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:     %ptr: type = ptr_type %Bar.ref.loc19_15 [concrete = constants.%ptr.f68]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %bar: %ptr.f68 = bind_name bar, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_public_data_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %ptr.f68: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.f68 [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: %pattern_type = binding_pattern bar [concrete]
// CHECK:STDOUT:     %bar.param_patt: %pattern_type = value_param_pattern %bar.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr.f68 = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:       %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Bar.ref.loc7: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr.loc7: type = ptr_type %Bar.ref.loc7 [concrete = constants.%ptr.f68]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr.f68 = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr.f68) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %foo_bar.patt: %pattern_type = binding_pattern foo_bar [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %bar.ref: %ptr.f68 = name_ref bar, %bar
// CHECK:STDOUT:   %.loc19_30: ref %Bar = deref %bar.ref
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc19_23: type = splice_block %ptr.loc19 [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:     %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Bar.ref.loc19: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:     %ptr.loc19: type = ptr_type %Bar.ref.loc19 [concrete = constants.%ptr.f68]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_bar: %ptr.f68 = bind_name foo_bar, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_inheritance_static.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bar1: type = class_type @Bar1 [concrete]
// CHECK:STDOUT:   %foo1.type.148: type = fn_type @foo1.1 [concrete]
// CHECK:STDOUT:   %foo1.8cd: %foo1.type.148 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar2: type = class_type @Bar2 [concrete]
// CHECK:STDOUT:   %foo1.type.0b8: type = fn_type @foo1.2 [concrete]
// CHECK:STDOUT:   %foo1.ba2: %foo1.type.0b8 = struct_value () [concrete]
// CHECK:STDOUT:   %foo2.type: type = fn_type @foo2 [concrete]
// CHECK:STDOUT:   %foo2: %foo2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar1 = %Bar1.decl
// CHECK:STDOUT:     .Bar2 = %Bar2.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar1.decl: type = class_decl @Bar1 [concrete = constants.%Bar1] {} {}
// CHECK:STDOUT:   %foo1.decl.c80: %foo1.type.148 = fn_decl @foo1.1 [concrete = constants.%foo1.8cd] {} {}
// CHECK:STDOUT:   %Bar2.decl: type = class_decl @Bar2 [concrete = constants.%Bar2] {} {}
// CHECK:STDOUT:   %foo1.decl.191: %foo1.type.0b8 = fn_decl @foo1.2 [concrete = constants.%foo1.ba2] {} {}
// CHECK:STDOUT:   %foo2.decl: %foo2.type = fn_decl @foo2 [concrete = constants.%foo2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Bar1.ref: type = name_ref Bar1, imports.%Bar1.decl [concrete = constants.%Bar1]
// CHECK:STDOUT:   %foo1.ref.loc8: %foo1.type.148 = name_ref foo1, imports.%foo1.decl.c80 [concrete = constants.%foo1.8cd]
// CHECK:STDOUT:   %foo1.call.loc8: init %empty_tuple.type = call %foo1.ref.loc8()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Bar2.ref.loc9: type = name_ref Bar2, imports.%Bar2.decl [concrete = constants.%Bar2]
// CHECK:STDOUT:   %foo1.ref.loc9: %foo1.type.0b8 = name_ref foo1, imports.%foo1.decl.191 [concrete = constants.%foo1.ba2]
// CHECK:STDOUT:   %foo1.call.loc9: init %empty_tuple.type = call %foo1.ref.loc9()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar2.ref.loc10: type = name_ref Bar2, imports.%Bar2.decl [concrete = constants.%Bar2]
// CHECK:STDOUT:   %foo2.ref: %foo2.type = name_ref foo2, imports.%foo2.decl [concrete = constants.%foo2]
// CHECK:STDOUT:   %foo2.call: init %empty_tuple.type = call %foo2.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_inheritance_pointers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar1: type = class_type @Bar1 [concrete]
// CHECK:STDOUT:   %ptr.f68: type = ptr_type %Bar1 [concrete]
// CHECK:STDOUT:   %pattern_type.3cc: type = pattern_type %ptr.f68 [concrete]
// CHECK:STDOUT:   %MyF1.type: type = fn_type @MyF1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyF1: %MyF1.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar2: type = class_type @Bar2 [concrete]
// CHECK:STDOUT:   %ptr.eca: type = ptr_type %Bar2 [concrete]
// CHECK:STDOUT:   %pattern_type.92a: type = pattern_type %ptr.eca [concrete]
// CHECK:STDOUT:   %MyF2.type: type = fn_type @MyF2 [concrete]
// CHECK:STDOUT:   %MyF2: %MyF2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar1 = %Bar1.decl
// CHECK:STDOUT:     .Bar2 = %Bar2.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar1.decl: type = class_decl @Bar1 [concrete = constants.%Bar1] {} {}
// CHECK:STDOUT:   %Bar2.decl: type = class_decl @Bar2 [concrete = constants.%Bar2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF1.decl: %MyF1.type = fn_decl @MyF1 [concrete = constants.%MyF1] {
// CHECK:STDOUT:     %bar.patt: %pattern_type.3cc = binding_pattern bar [concrete]
// CHECK:STDOUT:     %bar.param_patt: %pattern_type.3cc = value_param_pattern %bar.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr.f68 = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Bar1.ref: type = name_ref Bar1, imports.%Bar1.decl [concrete = constants.%Bar1]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar1.ref [concrete = constants.%ptr.f68]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr.f68 = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MyF2.decl: %MyF2.type = fn_decl @MyF2 [concrete = constants.%MyF2] {
// CHECK:STDOUT:     %bar.patt: %pattern_type.92a = binding_pattern bar [concrete]
// CHECK:STDOUT:     %bar.param_patt: %pattern_type.92a = value_param_pattern %bar.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr.eca = value_param call_param0
// CHECK:STDOUT:     %.loc19_23: type = splice_block %ptr [concrete = constants.%ptr.eca] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Bar2.ref: type = name_ref Bar2, imports.%Bar2.decl [concrete = constants.%Bar2]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar2.ref [concrete = constants.%ptr.eca]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr.eca = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF1(%bar.param: %ptr.f68);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF2(%bar.param: %ptr.eca) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MyF1.ref: %MyF1.type = name_ref MyF1, file.%MyF1.decl [concrete = constants.%MyF1]
// CHECK:STDOUT:   %bar.ref: %ptr.eca = name_ref bar, %bar
// CHECK:STDOUT:   %.loc19_33: %ptr.f68 = converted %bar.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %MyF1.call: init %empty_tuple.type = call %MyF1.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_dynamic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: <error> = binding_pattern bar [concrete]
// CHECK:STDOUT:     %bar.param_patt: <error> = value_param_pattern %bar.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc14: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.ref: <error> = name_ref Bar, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: <error> = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_to_inherit_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Bar [concrete]
// CHECK:STDOUT:   %struct_type.base.36d: type = struct_type {.base: %Bar} [concrete]
// CHECK:STDOUT:   %complete_type.fff: <witness> = complete_type_witness %struct_type.base.36d [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Bar.ref, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Bar} [concrete = constants.%struct_type.base.36d]
// CHECK:STDOUT:   %complete_type.loc9: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.fff]
// CHECK:STDOUT:   complete_type_witness = %complete_type.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .foo = <poisoned>
// CHECK:STDOUT:   extend %Bar.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = <poisoned>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.patt: <error> = binding_pattern bar [concrete]
// CHECK:STDOUT:     %bar.param_patt: <error> = value_param_pattern %bar.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc18: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Bar.ref: <error> = name_ref Bar, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: <error> = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: <error>);
// CHECK:STDOUT:
