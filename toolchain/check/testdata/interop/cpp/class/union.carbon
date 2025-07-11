// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/union.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/union.carbon

// ============================================================================
// Declaration
// ============================================================================

// --- declaration.h

union Bar;

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

union Bar {};

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

union Bar;
union Bar {};

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

union Bar {
 public:
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

union Bar {
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

union Bar {
 public:
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

union Bar {
 public:
  Bar* foo;
};

// --- fail_todo_import_public_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "public_static_data_member.h";

//@dump-sem-ir-begin
fn MyF(bar : Cpp.Bar*) {
  // CHECK:STDERR: fail_todo_import_public_data_member.carbon:[[@LINE+11]]:27: error: semantics TODO: `Unsupported: Declaration type Var` [SemanticsTodo]
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
// To inherit public
// ============================================================================

// --- to_inherit_public.h

union Bar {
 public:
  static auto foo() -> void;
};

// --- todo_fail_import_to_inherit_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "to_inherit_public.h";

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
union Bar {};

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
