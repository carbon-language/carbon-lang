// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
// EXTRA-ARGS: --clang-arg=--std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/concept.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/concept.carbon

// --- concept.h

struct Small {};
struct Large { int n; };

template<typename T> concept IsLarge = sizeof(T) > 1;

// --- use_concept.carbon

library "[[@TEST_NAME]]";

import Cpp library "concept.h";

class X(B:! bool) {
  fn Make() -> Self { return {}; }
}

//@dump-sem-ir-begin
var x1: X(Cpp.IsLarge(Cpp.Small)) = X(false).Make();
var x2: X(Cpp.IsLarge(Cpp.Large)) = X(true).Make();
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_concept.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %IsLarge.type: type = cpp_type_template_type IsLarge [concrete]
// CHECK:STDOUT:   %IsLarge.template: %IsLarge.type = struct_value () [concrete]
// CHECK:STDOUT:   %Small: type = class_type @Small [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %X.3b1: type = class_type @X, @X(%false) [concrete]
// CHECK:STDOUT:   %X.Make.type.ae8: type = fn_type @X.Make, @X(%false) [concrete]
// CHECK:STDOUT:   %X.Make.f03: %X.Make.type.ae8 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.b61: type = pattern_type %X.3b1 [concrete]
// CHECK:STDOUT:   %X.Make.specific_fn.a38: <specific function> = specific_function %X.Make.f03, @X.Make(%false) [concrete]
// CHECK:STDOUT:   %Large: type = class_type @Large [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %X.26c: type = class_type @X, @X(%true) [concrete]
// CHECK:STDOUT:   %X.Make.type.669: type = fn_type @X.Make, @X(%true) [concrete]
// CHECK:STDOUT:   %X.Make.8a2: %X.Make.type.669 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.63b: type = pattern_type %X.26c [concrete]
// CHECK:STDOUT:   %X.Make.specific_fn.17d: <specific function> = specific_function %X.Make.8a2, @X.Make(%true) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .IsLarge = %IsLarge.template
// CHECK:STDOUT:     .Small = %Small.decl
// CHECK:STDOUT:     .Large = %Large.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IsLarge.template: %IsLarge.type = struct_value () [concrete = constants.%IsLarge.template]
// CHECK:STDOUT:   %Small.decl: type = class_decl @Small [concrete = constants.%Small] {} {}
// CHECK:STDOUT:   %Large.decl: type = class_decl @Large [concrete = constants.%Large] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x1.patt: %pattern_type.b61 = ref_binding_pattern x1 [concrete]
// CHECK:STDOUT:     %x1.var_patt: %pattern_type.b61 = var_pattern %x1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x1.var: ref %X.3b1 = var %x1.var_patt [concrete]
// CHECK:STDOUT:   %.loc11: type = splice_block %X.loc11 [concrete = constants.%X.3b1] {
// CHECK:STDOUT:     %X.ref.loc11: %X.type = name_ref X, %X.decl [concrete = constants.%X.generic]
// CHECK:STDOUT:     %Cpp.ref.loc11_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IsLarge.ref.loc11: %IsLarge.type = name_ref IsLarge, imports.%IsLarge.template [concrete = constants.%IsLarge.template]
// CHECK:STDOUT:     %Cpp.ref.loc11_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Small.ref: type = name_ref Small, imports.%Small.decl [concrete = constants.%Small]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %X.loc11: type = class_type @X, @X(constants.%false) [concrete = constants.%X.3b1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x1: ref %X.3b1 = ref_binding x1, %x1.var [concrete = %x1.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x2.patt: %pattern_type.63b = ref_binding_pattern x2 [concrete]
// CHECK:STDOUT:     %x2.var_patt: %pattern_type.63b = var_pattern %x2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x2.var: ref %X.26c = var %x2.var_patt [concrete]
// CHECK:STDOUT:   %.loc12: type = splice_block %X.loc12 [concrete = constants.%X.26c] {
// CHECK:STDOUT:     %X.ref.loc12: %X.type = name_ref X, %X.decl [concrete = constants.%X.generic]
// CHECK:STDOUT:     %Cpp.ref.loc12_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %IsLarge.ref.loc12: %IsLarge.type = name_ref IsLarge, imports.%IsLarge.template [concrete = constants.%IsLarge.template]
// CHECK:STDOUT:     %Cpp.ref.loc12_23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Large.ref: type = name_ref Large, imports.%Large.decl [concrete = constants.%Large]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %X.loc12: type = class_type @X, @X(constants.%true) [concrete = constants.%X.26c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x2: ref %X.26c = ref_binding x2, %x2.var [concrete = %x2.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc11: %X.type = name_ref X, file.%X.decl [concrete = constants.%X.generic]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %X.loc11: type = class_type @X, @X(constants.%false) [concrete = constants.%X.3b1]
// CHECK:STDOUT:   %.loc11_45: %X.Make.type.ae8 = specific_constant @X.%X.Make.decl, @X(constants.%false) [concrete = constants.%X.Make.f03]
// CHECK:STDOUT:   %Make.ref.loc11: %X.Make.type.ae8 = name_ref Make, %.loc11_45 [concrete = constants.%X.Make.f03]
// CHECK:STDOUT:   %X.Make.specific_fn.loc11: <specific function> = specific_function %Make.ref.loc11, @X.Make(constants.%false) [concrete = constants.%X.Make.specific_fn.a38]
// CHECK:STDOUT:   %.loc11_1: ref %X.3b1 = splice_block file.%x1.var [concrete = file.%x1.var] {}
// CHECK:STDOUT:   %X.Make.call.loc11: init %X.3b1 to %.loc11_1 = call %X.Make.specific_fn.loc11()
// CHECK:STDOUT:   assign file.%x1.var, %X.Make.call.loc11
// CHECK:STDOUT:   %X.ref.loc12: %X.type = name_ref X, file.%X.decl [concrete = constants.%X.generic]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %X.loc12: type = class_type @X, @X(constants.%true) [concrete = constants.%X.26c]
// CHECK:STDOUT:   %.loc12_44: %X.Make.type.669 = specific_constant @X.%X.Make.decl, @X(constants.%true) [concrete = constants.%X.Make.8a2]
// CHECK:STDOUT:   %Make.ref.loc12: %X.Make.type.669 = name_ref Make, %.loc12_44 [concrete = constants.%X.Make.8a2]
// CHECK:STDOUT:   %X.Make.specific_fn.loc12: <specific function> = specific_function %Make.ref.loc12, @X.Make(constants.%true) [concrete = constants.%X.Make.specific_fn.17d]
// CHECK:STDOUT:   %.loc12_1: ref %X.26c = splice_block file.%x2.var [concrete = file.%x2.var] {}
// CHECK:STDOUT:   %X.Make.call.loc12: init %X.26c to %.loc12_1 = call %X.Make.specific_fn.loc12()
// CHECK:STDOUT:   assign file.%x2.var, %X.Make.call.loc12
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
