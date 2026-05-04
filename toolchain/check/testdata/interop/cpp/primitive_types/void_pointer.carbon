// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/void_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/void_pointer.carbon

// --- implicit_as_from_cpp_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.S*) {
  //@dump-sem-ir-begin
  let unused p: Cpp.void* = input;
  //@dump-sem-ir-end
}

// --- implicit_as_from_carbon_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: C*) {
  //@dump-sem-ir-begin
  let unused p: Cpp.void* = input;
  //@dump-sem-ir-end
}

// --- explicit_as_from_cpp_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.S*) {
  //@dump-sem-ir-begin
  let unused p: Cpp.void* = input as Cpp.void*;
  //@dump-sem-ir-end
}

// --- explicit_as_from_carbon_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: C*) {
  //@dump-sem-ir-begin
  let unused p: Cpp.void* = input as Cpp.void*;
  //@dump-sem-ir-end
}

// --- fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon:[[@LINE+7]]:26: error: cannot convert expression of type `Cpp.void*` to `Cpp.S*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused p: Cpp.S* = input as Cpp.S*;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon:[[@LINE+4]]:26: note: type `Cpp.void*` does not implement interface `Core.As(Cpp.S*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused p: Cpp.S* = input as Cpp.S*;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused p: Cpp.S* = input as Cpp.S*;
  //@dump-sem-ir-end
}

// --- fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon:[[@LINE+7]]:22: error: cannot convert expression of type `Cpp.void*` to `C*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused p: C* = input as C*;
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon:[[@LINE+4]]:22: note: type `Cpp.void*` does not implement interface `Core.As(C*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused p: C* = input as C*;
  // CHECK:STDERR:                      ^~~~~~~~~~~
  // CHECK:STDERR:
  let unused p: C* = input as C*;
  //@dump-sem-ir-end
}

// --- unsafe_as_from_void_pointer_to_cpp_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  let unused p: Cpp.S* = input unsafe as Cpp.S*;
  //@dump-sem-ir-end
}

// --- unsafe_as_from_void_pointer_to_carbon_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  let unused p: C* = input unsafe as C*;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- implicit_as_from_cpp_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.9fb: type = pattern_type %ptr.874 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.02d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.874)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.502: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.1b5(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.02e: %ptr.as.ImplicitAs.impl.Convert.type.502 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.429: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e02, @ptr.as.ImplicitAs.impl.1b5(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.31c: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.1b5(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.b9d: %ptr.as.ImplicitAs.impl.Convert.type.31c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.02d = facet_value %ptr.5c7, (%ImplicitAs.impl_witness.429) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.068: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.d36: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.068, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.9f6: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.b9d, @ptr.as.ImplicitAs.impl.Convert.1(%S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.86e: @ptr.as.ImplicitAs.impl.1b5.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.502) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.1b5.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.02e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e02 = impl_witness_table (%Core.import_ref.86e), @ptr.as.ImplicitAs.impl.1b5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.5c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.9fb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.5c7 = name_ref input, %input
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.874] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.874]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.d36 = impl_witness_access constants.%ImplicitAs.impl_witness.429, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.b9d]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%S) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.9f6]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.874 = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.874 = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.874 = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %p: %ptr.874 = value_binding p, %.loc10_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.9fb: type = pattern_type %ptr.874 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.02d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.874)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.502: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.1b5(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.02e: %ptr.as.ImplicitAs.impl.Convert.type.502 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f70: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e02, @ptr.as.ImplicitAs.impl.1b5(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.791: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.1b5(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.af8: %ptr.as.ImplicitAs.impl.Convert.type.791 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.02d = facet_value %ptr.31e, (%ImplicitAs.impl_witness.f70) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.1e1: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.41e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.1e1, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.a56: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.af8, @ptr.as.ImplicitAs.impl.Convert.1(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.86e: @ptr.as.ImplicitAs.impl.1b5.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.502) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.1b5.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.02e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e02 = impl_witness_table (%Core.import_ref.86e), @ptr.as.ImplicitAs.impl.1b5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.31e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.9fb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.31e = name_ref input, %input
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.874] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.874]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.41e = impl_witness_access constants.%ImplicitAs.impl_witness.f70, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.af8]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%C) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.a56]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.874 = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.874 = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.874 = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %p: %ptr.874 = value_binding p, %.loc10_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_cpp_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.9fb: type = pattern_type %ptr.874 [concrete]
// CHECK:STDOUT:   %As.type.d24: type = facet_type <@As, @As(%ptr.874)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.031604.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.738: %ImplicitAs.type.031604.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.628: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.738) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.5d5: %T.as_type.as.As.impl.Convert.type.628 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.02d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.874)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.502: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.1b5(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.02e: %ptr.as.ImplicitAs.impl.Convert.type.502 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.429: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e02, @ptr.as.ImplicitAs.impl.1b5(%S) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.02d = facet_value %ptr.5c7, (%ImplicitAs.impl_witness.429) [concrete]
// CHECK:STDOUT:   %As.impl_witness.58f: <witness> = impl_witness imports.%As.impl_witness_table.0dc, @T.as_type.as.As.impl(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.d89: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.e8b: %T.as_type.as.As.impl.Convert.type.d89 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d24 = facet_value %ptr.5c7, (%As.impl_witness.58f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2d8: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.874, %As.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %As.WithSelf.Convert.type.2d8, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.e8b, @T.as_type.as.As.impl.Convert(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.72f: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.628) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.5d5)]
// CHECK:STDOUT:   %As.impl_witness_table.0dc = impl_witness_table (%Core.import_ref.72f), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.86e: @ptr.as.ImplicitAs.impl.1b5.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.502) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.1b5.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.02e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e02 = impl_witness_table (%Core.import_ref.86e), @ptr.as.ImplicitAs.impl.1b5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.5c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.9fb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.5c7 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.874]
// CHECK:STDOUT:   %impl.elem0: %.187 = impl_witness_access constants.%As.impl_witness.58f, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.e8b]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.874, constants.%ImplicitAs.facet) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.874 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.874 = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.874 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.874] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.874]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.874 = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.9fb: type = pattern_type %ptr.874 [concrete]
// CHECK:STDOUT:   %As.type.d24: type = facet_type <@As, @As(%ptr.874)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.031604.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.738: %ImplicitAs.type.031604.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.628: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.738) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.5d5: %T.as_type.as.As.impl.Convert.type.628 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.02d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.874)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.502: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.1b5(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.02e: %ptr.as.ImplicitAs.impl.Convert.type.502 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f70: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e02, @ptr.as.ImplicitAs.impl.1b5(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.02d = facet_value %ptr.31e, (%ImplicitAs.impl_witness.f70) [concrete]
// CHECK:STDOUT:   %As.impl_witness.a9a: <witness> = impl_witness imports.%As.impl_witness_table.0dc, @T.as_type.as.As.impl(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.758: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.57a: %T.as_type.as.As.impl.Convert.type.758 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d24 = facet_value %ptr.31e, (%As.impl_witness.a9a) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.dc8: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.874, %As.facet) [concrete]
// CHECK:STDOUT:   %.85e: type = fn_type_with_self_type %As.WithSelf.Convert.type.dc8, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.57a, @T.as_type.as.As.impl.Convert(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.72f: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.628) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.5d5)]
// CHECK:STDOUT:   %As.impl_witness_table.0dc = impl_witness_table (%Core.import_ref.72f), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.86e: @ptr.as.ImplicitAs.impl.1b5.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.502) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.1b5.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.02e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e02 = impl_witness_table (%Core.import_ref.86e), @ptr.as.ImplicitAs.impl.1b5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.31e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.9fb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.31e = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.874]
// CHECK:STDOUT:   %impl.elem0: %.85e = impl_witness_access constants.%As.impl_witness.a9a, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.57a]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.874, constants.%ImplicitAs.facet) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.874 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.874 = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.874 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.874] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.874]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.874 = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc17_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc17_38: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc17_40: type = ptr_type %S.ref.loc17_38 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   %.loc17_32: %ptr.5c7 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_22: type = splice_block %ptr.loc17_22 [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc17_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc17_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc17_22: type = ptr_type %S.ref.loc17_20 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.5c7 = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.506 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc17_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc17_32: type = ptr_type %C.ref.loc17_31 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   %.loc17_28: %ptr.31e = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_18: type = splice_block %ptr.loc17_18 [concrete = constants.%ptr.31e] {
// CHECK:STDOUT:     %C.ref.loc17_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc17_18: type = ptr_type %C.ref.loc17_17 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.31e = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.259: type = pattern_type %ptr.5c7 [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.0ee: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.5c7)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.a8d: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.14a: %ptr.as.UnsafeAs.impl.Convert.type.a8d = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.d74: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.a63, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.6ce: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.699: %ptr.as.UnsafeAs.impl.Convert.type.6ce = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.0ee = facet_value %ptr.874, (%UnsafeAs.impl_witness.d74) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.dc0: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.5c7, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.1d8: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.dc0, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.699, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.19e: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.a8d) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.14a)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.a63 = impl_witness_table (%Core.import_ref.19e), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc10_45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc10_47: type = ptr_type %S.ref.loc10_45 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   %impl.elem0: %.1d8 = impl_witness_access constants.%UnsafeAs.impl_witness.d74, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.699]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%S) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.5c7 = call %bound_method.loc10_39.2(%input.ref)
// CHECK:STDOUT:   %.loc10_39.1: %ptr.5c7 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_39.2: %ptr.5c7 = converted %input.ref, %.loc10_39.1
// CHECK:STDOUT:   %.loc10_22: type = splice_block %ptr.loc10_22 [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc10_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc10_22: type = ptr_type %S.ref.loc10_20 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.5c7 = value_binding p, %.loc10_39.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.1ec: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.31e)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.a8d: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.14a: %ptr.as.UnsafeAs.impl.Convert.type.a8d = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.cb0: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.a63, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.475: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.b0e: %ptr.as.UnsafeAs.impl.Convert.type.475 = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.1ec = facet_value %ptr.874, (%UnsafeAs.impl_witness.cb0) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.ff0: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.31e, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.d87: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.ff0, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.b0e, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.19e: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.a8d) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.14a)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.a63 = impl_witness_table (%Core.import_ref.19e), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.506 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc10_38: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc10_39: type = ptr_type %C.ref.loc10_38 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   %impl.elem0: %.d87 = impl_witness_access constants.%UnsafeAs.impl_witness.cb0, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.b0e]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%C) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.31e = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.31e = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.31e = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10_18 [concrete = constants.%ptr.31e] {
// CHECK:STDOUT:     %C.ref.loc10_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_18: type = ptr_type %C.ref.loc10_17 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.31e = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
