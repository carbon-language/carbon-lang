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
// CHECK:STDOUT:   %ptr.9e9: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.c0f: type = pattern_type %ptr.838 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.919: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.838)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.bdb: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.984(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.7e2: %ptr.as.ImplicitAs.impl.Convert.type.bdb = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e3b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfc, @ptr.as.ImplicitAs.impl.984(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.5da: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.984(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.9d2: %ptr.as.ImplicitAs.impl.Convert.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1d5: %ImplicitAs.type.919 = facet_value %ptr.9e9, (%ImplicitAs.impl_witness.e3b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.076: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.838, %ImplicitAs.facet.1d5) [concrete]
// CHECK:STDOUT:   %.d01: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.076, %ImplicitAs.facet.1d5 [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.222: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.9d2, @ptr.as.ImplicitAs.impl.Convert.1(%S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.a68: @ptr.as.ImplicitAs.impl.984.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.bdb) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.984.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.7e2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfc = impl_witness_table (%Core.import_ref.a68), @ptr.as.ImplicitAs.impl.984 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.9e9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.c0f = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.9e9 = name_ref input, %input
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.838] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.838]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.d01 = impl_witness_access constants.%ImplicitAs.impl_witness.e3b, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.9d2]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%S) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.222]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.838 = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.838 = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.838 = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %p: %ptr.838 = value_binding p, %.loc10_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.1d8: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.c0f: type = pattern_type %ptr.838 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.919: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.838)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.bdb: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.984(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.7e2: %ptr.as.ImplicitAs.impl.Convert.type.bdb = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.253: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfc, @ptr.as.ImplicitAs.impl.984(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.d28: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.984(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.9a0: %ptr.as.ImplicitAs.impl.Convert.type.d28 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7ca: %ImplicitAs.type.919 = facet_value %ptr.1d8, (%ImplicitAs.impl_witness.253) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.15e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.838, %ImplicitAs.facet.7ca) [concrete]
// CHECK:STDOUT:   %.e60: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.15e, %ImplicitAs.facet.7ca [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.609: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.9a0, @ptr.as.ImplicitAs.impl.Convert.1(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a68: @ptr.as.ImplicitAs.impl.984.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.bdb) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.984.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.7e2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfc = impl_witness_table (%Core.import_ref.a68), @ptr.as.ImplicitAs.impl.984 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.1d8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.c0f = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.1d8 = name_ref input, %input
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.838] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.838]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.e60 = impl_witness_access constants.%ImplicitAs.impl_witness.253, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.9a0]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%C) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.609]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.838 = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.838 = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.838 = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %p: %ptr.838 = value_binding p, %.loc10_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_cpp_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.9e9: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.c0f: type = pattern_type %ptr.838 [concrete]
// CHECK:STDOUT:   %As.type.8e0: type = facet_type <@As, @As(%ptr.838)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0b443a.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.2dd: %ImplicitAs.type.0b443a.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.c4b: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.2dd) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.791: %T.as_type.as.As.impl.Convert.type.c4b = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.919: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.838)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.bdb: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.984(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.7e2: %ptr.as.ImplicitAs.impl.Convert.type.bdb = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e3b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfc, @ptr.as.ImplicitAs.impl.984(%S) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1d5: %ImplicitAs.type.919 = facet_value %ptr.9e9, (%ImplicitAs.impl_witness.e3b) [concrete]
// CHECK:STDOUT:   %As.impl_witness.72f: <witness> = impl_witness imports.%As.impl_witness_table.dad, @T.as_type.as.As.impl(%ptr.838, %ImplicitAs.facet.1d5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.db3: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.838, %ImplicitAs.facet.1d5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.35b: %T.as_type.as.As.impl.Convert.type.db3 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.8e0 = facet_value %ptr.9e9, (%As.impl_witness.72f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.b23: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.838, %As.facet) [concrete]
// CHECK:STDOUT:   %.b17: type = fn_type_with_self_type %As.WithSelf.Convert.type.b23, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.35b, @T.as_type.as.As.impl.Convert(%ptr.838, %ImplicitAs.facet.1d5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.6a9: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.c4b) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.791)]
// CHECK:STDOUT:   %As.impl_witness_table.dad = impl_witness_table (%Core.import_ref.6a9), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a68: @ptr.as.ImplicitAs.impl.984.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.bdb) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.984.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.7e2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfc = impl_witness_table (%Core.import_ref.a68), @ptr.as.ImplicitAs.impl.984 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.9e9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.c0f = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.9e9 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.838]
// CHECK:STDOUT:   %impl.elem0: %.b17 = impl_witness_access constants.%As.impl_witness.72f, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.35b]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.838, constants.%ImplicitAs.facet.1d5) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.838 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.838 = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.838 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.838] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.838]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.838 = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.ad5: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.c0f: type = pattern_type %ptr.838 [concrete]
// CHECK:STDOUT:   %As.type.8e0: type = facet_type <@As, @As(%ptr.838)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0b443a.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.2dd: %ImplicitAs.type.0b443a.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.c4b: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.2dd) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.791: %T.as_type.as.As.impl.Convert.type.c4b = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.919: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.838)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.bdb: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.984(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.7e2: %ptr.as.ImplicitAs.impl.Convert.type.bdb = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.163: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bfc, @ptr.as.ImplicitAs.impl.984(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.bb0: %ImplicitAs.type.919 = facet_value %ptr.ad5, (%ImplicitAs.impl_witness.163) [concrete]
// CHECK:STDOUT:   %As.impl_witness.e6f: <witness> = impl_witness imports.%As.impl_witness_table.dad, @T.as_type.as.As.impl(%ptr.838, %ImplicitAs.facet.bb0) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.364: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.838, %ImplicitAs.facet.bb0) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.bc1: %T.as_type.as.As.impl.Convert.type.364 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.8e0 = facet_value %ptr.ad5, (%As.impl_witness.e6f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.aa3: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.838, %As.facet) [concrete]
// CHECK:STDOUT:   %.e70: type = fn_type_with_self_type %As.WithSelf.Convert.type.aa3, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.bc1, @T.as_type.as.As.impl.Convert(%ptr.838, %ImplicitAs.facet.bb0) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.6a9: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.c4b) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.791)]
// CHECK:STDOUT:   %As.impl_witness_table.dad = impl_witness_table (%Core.import_ref.6a9), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a68: @ptr.as.ImplicitAs.impl.984.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.bdb) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.984.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.7e2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bfc = impl_witness_table (%Core.import_ref.a68), @ptr.as.ImplicitAs.impl.984 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.ad5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.c0f = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.ad5 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.838]
// CHECK:STDOUT:   %impl.elem0: %.e70 = impl_witness_access constants.%As.impl_witness.e6f, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.bc1]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.838, constants.%ImplicitAs.facet.bb0) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.838 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.838 = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.838 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.838] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.838]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.838 = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.9e9: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %ptr.9e9 [concrete]
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
// CHECK:STDOUT: fn @F(%input.param: %ptr.838) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.761 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.838 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc17_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc17_38: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc17_40: type = ptr_type %S.ref.loc17_38 [concrete = constants.%ptr.9e9]
// CHECK:STDOUT:   %.loc17_32: %ptr.9e9 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_22: type = splice_block %ptr.loc17_22 [concrete = constants.%ptr.9e9] {
// CHECK:STDOUT:     %Cpp.ref.loc17_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc17_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc17_22: type = ptr_type %S.ref.loc17_20 [concrete = constants.%ptr.9e9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.9e9 = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.806: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.b5c: type = pattern_type %ptr.806 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.838) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.b5c = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.838 = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc17_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc17_32: type = ptr_type %C.ref.loc17_31 [concrete = constants.%ptr.806]
// CHECK:STDOUT:   %.loc17_28: %ptr.806 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_18: type = splice_block %ptr.loc17_18 [concrete = constants.%ptr.806] {
// CHECK:STDOUT:     %C.ref.loc17_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc17_18: type = ptr_type %C.ref.loc17_17 [concrete = constants.%ptr.806]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.806 = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.9e9: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %ptr.9e9 [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.e37: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.9e9)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.ed0: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.8b4: %ptr.as.UnsafeAs.impl.Convert.type.ed0 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.4bb: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.546, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.83e: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.a6b: %ptr.as.UnsafeAs.impl.Convert.type.83e = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.e37 = facet_value %ptr.838, (%UnsafeAs.impl_witness.4bb) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.96c: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.9e9, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.7a8: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.96c, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.a6b, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.bce: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.ed0) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.8b4)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.546 = impl_witness_table (%Core.import_ref.bce), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.838) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.761 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.838 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc10_45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc10_47: type = ptr_type %S.ref.loc10_45 [concrete = constants.%ptr.9e9]
// CHECK:STDOUT:   %impl.elem0: %.7a8 = impl_witness_access constants.%UnsafeAs.impl_witness.4bb, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.a6b]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%S) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.9e9 = call %bound_method.loc10_39.2(%input.ref)
// CHECK:STDOUT:   %.loc10_39.1: %ptr.9e9 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_39.2: %ptr.9e9 = converted %input.ref, %.loc10_39.1
// CHECK:STDOUT:   %.loc10_22: type = splice_block %ptr.loc10_22 [concrete = constants.%ptr.9e9] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc10_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc10_22: type = ptr_type %S.ref.loc10_20 [concrete = constants.%ptr.9e9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.9e9 = value_binding p, %.loc10_39.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.838: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.d10: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.c97: type = pattern_type %ptr.d10 [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.d54: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.d10)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.ed0: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.8b4: %ptr.as.UnsafeAs.impl.Convert.type.ed0 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.676: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.546, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.dd5: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.3f5: %ptr.as.UnsafeAs.impl.Convert.type.dd5 = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.d54 = facet_value %ptr.838, (%UnsafeAs.impl_witness.676) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.802: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.d10, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.939: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.802, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.3f5, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bce: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.ed0) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.8b4)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.546 = impl_witness_table (%Core.import_ref.bce), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.838) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.c97 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.838 = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc10_38: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc10_39: type = ptr_type %C.ref.loc10_38 [concrete = constants.%ptr.d10]
// CHECK:STDOUT:   %impl.elem0: %.939 = impl_witness_access constants.%UnsafeAs.impl_witness.676, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.3f5]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%C) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.d10 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.d10 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.d10 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10_18 [concrete = constants.%ptr.d10] {
// CHECK:STDOUT:     %C.ref.loc10_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_18: type = ptr_type %C.ref.loc10_17 [concrete = constants.%ptr.d10]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.d10 = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
