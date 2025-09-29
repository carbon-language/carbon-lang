// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/class.carbon

// ============================================================================
// Declaration
// ============================================================================

// --- declaration.h

class Bar;

// --- import_declaration.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.Bar*);
//@dump-sem-ir-end

// --- fail_use_declaration_as_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration.h";

fn MyF() {
  // CHECK:STDERR: fail_use_declaration_as_definition.carbon:[[@LINE+8]]:12: error: binding pattern has incomplete type `Bar` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   var bar: Cpp.Bar;
  // CHECK:STDERR:            ^~~~~~~
  // CHECK:STDERR: fail_use_declaration_as_definition.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./declaration.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Bar;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  var bar: Cpp.Bar;
}

// ============================================================================
// Definition
// ============================================================================

// --- definition.h

class Bar {};

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

class Bar;
class Bar {};

// --- import_declaration_and_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration_and_definition.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.Bar*);
//@dump-sem-ir-end

// ============================================================================
// Static member function
// ============================================================================

// --- static_member_function.h

class Bar {
 public:
  static auto foo() -> void;
};

// --- import_static_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "static_member_function.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.Bar.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Static data member
// ============================================================================

// --- static_data_member.h

class Bar {
 public:
  static Bar* _Nonnull foo;
};

// --- import_static_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "static_data_member.h";

fn MyF() {
  //@dump-sem-ir-begin
  let bar: Cpp.Bar* = Cpp.Bar.foo;
  //@dump-sem-ir-end
}

// ============================================================================
// Data member
// ============================================================================

// --- data_member.h

class Bar {
 public:
  Bar* _Nonnull foo;
};

// --- import_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "data_member.h";

//@dump-sem-ir-begin
fn MyF(bar : Cpp.Bar*) {
  let foo_bar: Cpp.Bar* = bar->foo;
}
//@dump-sem-ir-end

// ============================================================================
// Dynamic
// ============================================================================

// --- dynamic.h

class Bar {
 public:
  virtual ~Bar();
};

// --- import_dynamic.carbon

library "[[@TEST_NAME]]";

import Cpp library "dynamic.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.Bar*);
//@dump-sem-ir-end

// ============================================================================
// To inherit public
// ============================================================================

// --- to_inherit_public.h

class Bar {
 public:
  static auto foo() -> void;
};

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
// Template
// ============================================================================

// --- template.h

template<typename T>
class Bar {};

// --- fail_todo_import_template.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_import_template.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./template.h:3:7: error: semantics TODO: `Unsupported: Declaration type ClassTemplate` [SemanticsTodo]
// CHECK:STDERR: class Bar {};
// CHECK:STDERR:       ^
import Cpp library "template.h";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_import_template.carbon:[[@LINE+4]]:13: note: in `Cpp` name lookup for `Bar` [InCppNameLookup]
// CHECK:STDERR: fn MyF(bar: Cpp.Bar*);
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
fn MyF(bar: Cpp.Bar*);
//@dump-sem-ir-end

// CHECK:STDOUT: --- import_declaration.carbon
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
// CHECK:STDOUT:       %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
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
// CHECK:STDOUT:       %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_static_member_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %.c07: type = cpp_overload_set_type @Bar.foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c07 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.foo.type: type = fn_type @Bar.foo [concrete]
// CHECK:STDOUT:   %Bar.foo: %Bar.foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:   %.617: %.c07 = cpp_overload_set_value @Bar.foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Bar.foo.decl: %Bar.foo.type = fn_decl @Bar.foo [concrete = constants.%Bar.foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %foo.ref: %.c07 = name_ref foo, imports.%.617 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Bar.foo.call: init %empty_tuple.type = call imports.%Bar.foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_static_data_member.carbon
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
// CHECK:STDOUT:   %foo.var: ref %ptr.f68 = var %foo.var_patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bar.patt: %pattern_type = binding_pattern bar [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar.ref.loc8_26: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %foo.ref: ref %ptr.f68 = name_ref foo, imports.%foo.var
// CHECK:STDOUT:   %.loc8_19: type = splice_block %ptr [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:     %Cpp.ref.loc8_12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Bar.ref.loc8_15: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:     %ptr: type = ptr_type %Bar.ref.loc8_15 [concrete = constants.%ptr.f68]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_30: %ptr.f68 = bind_value %foo.ref
// CHECK:STDOUT:   %bar: %ptr.f68 = bind_name bar, %.loc8_30
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_data_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bar: type = class_type @Bar [concrete]
// CHECK:STDOUT:   %ptr.f68: type = ptr_type %Bar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.f68 [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.elem: type = unbound_element_type %Bar, %ptr.f68 [concrete]
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
// CHECK:STDOUT:   %.loc8_30.1: ref %Bar = deref %bar.ref
// CHECK:STDOUT:   %foo.ref: %Bar.elem = name_ref foo, @Bar.%.1 [concrete = @Bar.%.1]
// CHECK:STDOUT:   %.loc8_30.2: ref %ptr.f68 = class_element_access %.loc8_30.1, element0
// CHECK:STDOUT:   %.loc8_23: type = splice_block %ptr.loc8 [concrete = constants.%ptr.f68] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Bar.ref.loc8: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:     %ptr.loc8: type = ptr_type %Bar.ref.loc8 [concrete = constants.%ptr.f68]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_30.3: %ptr.f68 = bind_value %.loc8_30.2
// CHECK:STDOUT:   %foo_bar: %ptr.f68 = bind_name foo_bar, %.loc8_30.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_dynamic.carbon
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
// CHECK:STDOUT:       %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:       %ptr: type = ptr_type %Bar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = bind_name bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
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
// CHECK:STDOUT:   %.c07: type = cpp_overload_set_type @Bar.foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c07 = struct_value () [concrete]
// CHECK:STDOUT:   %Bar.foo.type: type = fn_type @Bar.foo [concrete]
// CHECK:STDOUT:   %Bar.foo: %Bar.foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Bar = %Bar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bar.decl: type = class_decl @Bar [concrete = constants.%Bar] {} {}
// CHECK:STDOUT:   %.617: %.c07 = cpp_overload_set_value @Bar.foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Bar.foo.decl: %Bar.foo.type = fn_decl @Bar.foo [concrete = constants.%Bar.foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Bar.ref: type = name_ref Bar, imports.%Bar.decl [concrete = constants.%Bar]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Bar.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.36d [concrete = constants.%complete_type.fff]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %foo.ref: %.c07 = name_ref foo, imports.%.617 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Bar.foo.call: init %empty_tuple.type = call imports.%Bar.foo.decl()
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
// CHECK:STDOUT:     %.loc15: type = splice_block %ptr [concrete = <error>] {
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
