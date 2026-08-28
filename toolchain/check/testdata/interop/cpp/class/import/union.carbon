// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/union.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/union.carbon

// ============================================================================
// Declaration
// ============================================================================

// --- declaration.h

union DeclBar;

// --- import_declaration.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.DeclBar*);
//@dump-sem-ir-end

// --- fail_use_declaration_as_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration.h";

fn MyF() {
  // CHECK:STDERR: fail_use_declaration_as_definition.carbon:[[@LINE+4]]:19-29: error: binding pattern has incomplete type `DeclBar` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR: included from: fail_use_declaration_as_definition.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./declaration.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  var unused bar: Cpp.DeclBar;
}

// ============================================================================
// Definition
// ============================================================================

// --- definition.h

union DefBar {};

// --- import_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "definition.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.DefBar*);
//@dump-sem-ir-end

// ============================================================================
// Declaration and definition
// ============================================================================

// --- declaration_and_definition.h

union DeclDefBar;
union DeclDefBar {};

// --- import_declaration_and_definition.carbon

library "[[@TEST_NAME]]";

import Cpp library "declaration_and_definition.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.DeclDefBar*);
//@dump-sem-ir-end

// ============================================================================
// Static member function
// ============================================================================

// --- static_member_function.h

union StaticMemberFnBar {
 public:
  static auto foo() -> void;
};

// --- import_static_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "static_member_function.h";

fn MyF() {
  //@dump-sem-ir-begin
  Cpp.StaticMemberFnBar.foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Static data member
// ============================================================================

// --- static_data_member.h

union StaticDataMemberBar {
 public:
  static StaticDataMemberBar* _Nonnull foo;
};

// --- import_static_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "static_data_member.h";

fn MyF() {
  //@dump-sem-ir-begin
  let unused bar: Cpp.StaticDataMemberBar* = Cpp.StaticDataMemberBar.foo;
  //@dump-sem-ir-end
}

// ============================================================================
// Data member
// ============================================================================

// --- data_member.h

union DataMemberBar {
 public:
  DataMemberBar* _Nonnull foo;
};

// --- import_data_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "data_member.h";

//@dump-sem-ir-begin
fn MyF(bar : Cpp.DataMemberBar*) {
  let unused foo_bar: Cpp.DataMemberBar* = bar->foo;
}
//@dump-sem-ir-end

// ============================================================================
// Template
// ============================================================================

// --- template.h

struct X {};

template<typename T>
union TemplateBar {};

// --- import_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "template.h";

//@dump-sem-ir-begin
fn MyF(bar: Cpp.TemplateBar(Cpp.X)*);
//@dump-sem-ir-end

// CHECK:STDOUT: --- import_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeclBar: type = class_type @DeclBar [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeclBar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %bar.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DeclBar = %DeclBar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeclBar.decl: type = class_decl @DeclBar [concrete = constants.%DeclBar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.param_patt: %pattern_type = value_param_pattern [concrete = constants.%bar.param_patt]
// CHECK:STDOUT:     %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete = constants.%bar.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DeclBar.ref: type = name_ref DeclBar, imports.%DeclBar.decl [concrete = constants.%DeclBar]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeclBar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = wrapper_binding bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DefBar: type = class_type @DefBar [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DefBar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %bar.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DefBar = %DefBar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefBar.decl: type = class_decl @DefBar [concrete = constants.%DefBar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.param_patt: %pattern_type = value_param_pattern [concrete = constants.%bar.param_patt]
// CHECK:STDOUT:     %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete = constants.%bar.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DefBar.ref: type = name_ref DefBar, imports.%DefBar.decl [concrete = constants.%DefBar]
// CHECK:STDOUT:       %ptr: type = ptr_type %DefBar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = wrapper_binding bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_declaration_and_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DeclDefBar: type = class_type @DeclDefBar [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %DeclDefBar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %bar.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DeclDefBar = %DeclDefBar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DeclDefBar.decl: type = class_decl @DeclDefBar [concrete = constants.%DeclDefBar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.param_patt: %pattern_type = value_param_pattern [concrete = constants.%bar.param_patt]
// CHECK:STDOUT:     %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete = constants.%bar.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DeclDefBar.ref: type = name_ref DeclDefBar, imports.%DeclDefBar.decl [concrete = constants.%DeclDefBar]
// CHECK:STDOUT:       %ptr: type = ptr_type %DeclDefBar.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = wrapper_binding bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_static_member_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %StaticMemberFnBar: type = class_type @StaticMemberFnBar [concrete]
// CHECK:STDOUT:   %StaticMemberFnBar.foo.cpp_overload_set.type: type = cpp_overload_set_type @StaticMemberFnBar.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMemberFnBar.foo.cpp_overload_set.value: %StaticMemberFnBar.foo.cpp_overload_set.type = cpp_overload_set_value @StaticMemberFnBar.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %StaticMemberFnBar.foo.type: type = fn_type @StaticMemberFnBar.foo [concrete]
// CHECK:STDOUT:   %StaticMemberFnBar.foo: %StaticMemberFnBar.foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticMemberFnBar = %StaticMemberFnBar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMemberFnBar.decl: type = class_decl @StaticMemberFnBar [concrete = constants.%StaticMemberFnBar] {} {}
// CHECK:STDOUT:   %StaticMemberFnBar.foo.cpp_overload_set.value: %StaticMemberFnBar.foo.cpp_overload_set.type = cpp_overload_set_value @StaticMemberFnBar.foo.cpp_overload_set [concrete = constants.%StaticMemberFnBar.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMemberFnBar.foo.decl: %StaticMemberFnBar.foo.type = fn_decl @StaticMemberFnBar.foo [concrete = constants.%StaticMemberFnBar.foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticMemberFnBar.ref: type = name_ref StaticMemberFnBar, imports.%StaticMemberFnBar.decl [concrete = constants.%StaticMemberFnBar]
// CHECK:STDOUT:   %foo.ref: %StaticMemberFnBar.foo.cpp_overload_set.type = name_ref foo, imports.%StaticMemberFnBar.foo.cpp_overload_set.value [concrete = constants.%StaticMemberFnBar.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %StaticMemberFnBar.foo.call: init %empty_tuple.type = call imports.%StaticMemberFnBar.foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_static_data_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StaticDataMemberBar: type = class_type @StaticDataMemberBar [concrete]
// CHECK:STDOUT:   %ptr.a89: type = ptr_type %StaticDataMemberBar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.a89 [concrete]
// CHECK:STDOUT:   %bar.patt: %pattern_type = value_binding_pattern bar [concrete]
// CHECK:STDOUT:   %foo.patt: %pattern_type = ref_binding_pattern foo [concrete]
// CHECK:STDOUT:   %foo.var_patt: %pattern_type = var_pattern %foo.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .StaticDataMemberBar = %StaticDataMemberBar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticDataMemberBar.decl: type = class_decl @StaticDataMemberBar [concrete = constants.%StaticDataMemberBar] {} {}
// CHECK:STDOUT:   %foo.patt: %pattern_type = ref_binding_pattern foo [concrete = constants.%foo.patt]
// CHECK:STDOUT:   %foo.var_patt: %pattern_type = var_pattern %foo.patt [concrete = constants.%foo.var_patt]
// CHECK:STDOUT:   %foo.var: ref %ptr.a89 = var_storage %foo.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_46: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %StaticDataMemberBar.ref.loc8_49: type = name_ref StaticDataMemberBar, imports.%StaticDataMemberBar.decl [concrete = constants.%StaticDataMemberBar]
// CHECK:STDOUT:   %foo.ref: ref %ptr.a89 = name_ref foo, imports.%foo.var [concrete = imports.%foo.var]
// CHECK:STDOUT:   %.loc8_69: %ptr.a89 = acquire_value %foo.ref
// CHECK:STDOUT:   %.loc8_42: type = splice_block %ptr [concrete = constants.%ptr.a89] {
// CHECK:STDOUT:     %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %StaticDataMemberBar.ref.loc8_22: type = name_ref StaticDataMemberBar, imports.%StaticDataMemberBar.decl [concrete = constants.%StaticDataMemberBar]
// CHECK:STDOUT:     %ptr: type = ptr_type %StaticDataMemberBar.ref.loc8_22 [concrete = constants.%ptr.a89]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %bar: %ptr.a89 = wrapper_binding bar, %.loc8_69
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %bar.patt: %pattern_type = value_binding_pattern bar [concrete = constants.%bar.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_data_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DataMemberBar: type = class_type @DataMemberBar [concrete]
// CHECK:STDOUT:   %ptr.ec1: type = ptr_type %DataMemberBar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.ec1 [concrete]
// CHECK:STDOUT:   %bar.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo_bar.patt: %pattern_type = value_binding_pattern foo_bar [concrete]
// CHECK:STDOUT:   %DataMemberBar.elem: type = unbound_element_type %DataMemberBar, %ptr.ec1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DataMemberBar = %DataMemberBar.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DataMemberBar.decl: type = class_decl @DataMemberBar [concrete = constants.%DataMemberBar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.param_patt: %pattern_type = value_param_pattern [concrete = constants.%bar.param_patt]
// CHECK:STDOUT:     %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete = constants.%bar.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr.ec1 = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.ec1] {
// CHECK:STDOUT:       %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DataMemberBar.ref.loc7: type = name_ref DataMemberBar, imports.%DataMemberBar.decl [concrete = constants.%DataMemberBar]
// CHECK:STDOUT:       %ptr.loc7: type = ptr_type %DataMemberBar.ref.loc7 [concrete = constants.%ptr.ec1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr.ec1 = wrapper_binding bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr.ec1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %bar.ref: %ptr.ec1 = name_ref bar, %bar
// CHECK:STDOUT:   %.loc8_47.1: ref %DataMemberBar = deref %bar.ref
// CHECK:STDOUT:   %foo.ref: %DataMemberBar.elem = name_ref foo, @DataMemberBar.%.1 [concrete = @DataMemberBar.%.1]
// CHECK:STDOUT:   %.loc8_47.2: ref %ptr.ec1 = class_element_access %.loc8_47.1, element0
// CHECK:STDOUT:   %.loc8_47.3: %ptr.ec1 = acquire_value %.loc8_47.2
// CHECK:STDOUT:   %.loc8_40: type = splice_block %ptr.loc8 [concrete = constants.%ptr.ec1] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DataMemberBar.ref.loc8: type = name_ref DataMemberBar, imports.%DataMemberBar.decl [concrete = constants.%DataMemberBar]
// CHECK:STDOUT:     %ptr.loc8: type = ptr_type %DataMemberBar.ref.loc8 [concrete = constants.%ptr.ec1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_bar: %ptr.ec1 = wrapper_binding foo_bar, %.loc8_47.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %foo_bar.patt: %pattern_type = value_binding_pattern foo_bar [concrete = constants.%foo_bar.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TemplateBar.type: type = cpp_type_template_type TemplateBar [concrete]
// CHECK:STDOUT:   %TemplateBar.template: %TemplateBar.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %TemplateBar: type = class_type @TemplateBar [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %TemplateBar [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %bar.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete]
// CHECK:STDOUT:   %MyF.type: type = fn_type @MyF [concrete]
// CHECK:STDOUT:   %MyF: %MyF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TemplateBar = %TemplateBar.template
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TemplateBar.template: %TemplateBar.type = struct_value () [concrete = constants.%TemplateBar.template]
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %TemplateBar.decl: type = class_decl @TemplateBar [concrete = constants.%TemplateBar] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %MyF.decl: %MyF.type = fn_decl @MyF [concrete = constants.%MyF] {
// CHECK:STDOUT:     %bar.param_patt: %pattern_type = value_param_pattern [concrete = constants.%bar.param_patt]
// CHECK:STDOUT:     %bar.patt: %pattern_type = wrapper_binding_pattern bar, %bar.param_patt [concrete = constants.%bar.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bar.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Cpp.ref.loc7_13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %TemplateBar.ref: %TemplateBar.type = name_ref TemplateBar, imports.%TemplateBar.template [concrete = constants.%TemplateBar.template]
// CHECK:STDOUT:       %Cpp.ref.loc7_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %X.ref: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %ptr: type = ptr_type imports.%TemplateBar.decl [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %bar: %ptr = wrapper_binding bar, %bar.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyF(%bar.param: %ptr);
// CHECK:STDOUT:
