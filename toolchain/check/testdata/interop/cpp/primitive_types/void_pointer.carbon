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
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon:[[@LINE+3]]:26-40: error: cannot convert expression of type `Cpp.void*` to `Cpp.S*` with `as` [ConversionFailure]
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon:[[@LINE+2]]:26-40: note: type `Cpp.void*` does not implement interface `Core.As(Cpp.S*)` [MissingImplInMemberAccessInContext]
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
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon:[[@LINE+3]]:22-32: error: cannot convert expression of type `Cpp.void*` to `C*` with `as` [ConversionFailure]
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon:[[@LINE+2]]:22-32: note: type `Cpp.void*` does not implement interface `Core.As(C*)` [MissingImplInMemberAccessInContext]
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
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.712: type = pattern_type %ptr.00a [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.712 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.c59: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.00a)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.8e0: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.992(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.a18: %ptr.as.ImplicitAs.impl.Convert.type.8e0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.033: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c14, @ptr.as.ImplicitAs.impl.992(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.b93: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.992(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.313: %ptr.as.ImplicitAs.impl.Convert.type.b93 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9ea: %ImplicitAs.type.c59 = facet_value %ptr.037, (%ImplicitAs.impl_witness.033) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.371: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.00a, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT:   %.00d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.371, %ImplicitAs.facet.9ea [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.642: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.313, @ptr.as.ImplicitAs.impl.Convert.1(%S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.47d: @ptr.as.ImplicitAs.impl.992.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.8e0) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.992.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.a18)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c14 = impl_witness_table (%Core.import_ref.47d), @ptr.as.ImplicitAs.impl.992 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.037) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %input.ref: %ptr.037 = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.00d = impl_witness_access constants.%ImplicitAs.impl_witness.033, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.313]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%S) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.642]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.00a = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.00a = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.00a = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.00a] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.00a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.00a = wrapper_binding p, %.loc10_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.712 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.712: type = pattern_type %ptr.00a [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.712 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.c59: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.00a)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.8e0: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.992(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.a18: %ptr.as.ImplicitAs.impl.Convert.type.8e0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.37b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c14, @ptr.as.ImplicitAs.impl.992(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.c38: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.992(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.458: %ptr.as.ImplicitAs.impl.Convert.type.c38 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.08f: %ImplicitAs.type.c59 = facet_value %ptr.6b6, (%ImplicitAs.impl_witness.37b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.620: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.00a, %ImplicitAs.facet.08f) [concrete]
// CHECK:STDOUT:   %.d86: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.620, %ImplicitAs.facet.08f [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.aae: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.458, @ptr.as.ImplicitAs.impl.Convert.1(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.47d: @ptr.as.ImplicitAs.impl.992.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.8e0) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.992.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.a18)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c14 = impl_witness_table (%Core.import_ref.47d), @ptr.as.ImplicitAs.impl.992 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.6b6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %input.ref: %ptr.6b6 = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.d86 = impl_witness_access constants.%ImplicitAs.impl_witness.37b, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.458]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%C) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.aae]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.00a = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.00a = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.00a = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.00a] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.00a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.00a = wrapper_binding p, %.loc10_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.712 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_cpp_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.712: type = pattern_type %ptr.00a [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.712 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %As.type.ffb: type = facet_type <@As, @As(%ptr.00a)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa489.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.294: %ImplicitAs.type.3aa489.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.1a4: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.294) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.0c3: %T.as_type.as.As.impl.Convert.type.1a4 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c59: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.00a)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.8e0: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.992(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.a18: %ptr.as.ImplicitAs.impl.Convert.type.8e0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.033: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c14, @ptr.as.ImplicitAs.impl.992(%S) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9ea: %ImplicitAs.type.c59 = facet_value %ptr.037, (%ImplicitAs.impl_witness.033) [concrete]
// CHECK:STDOUT:   %As.impl_witness.cac: <witness> = impl_witness imports.%As.impl_witness_table.10e, @T.as_type.as.As.impl(%ptr.00a, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.4ae: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.00a, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.189: %T.as_type.as.As.impl.Convert.type.4ae = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.ffb = facet_value %ptr.037, (%As.impl_witness.cac) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.565: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.00a, %As.facet) [concrete]
// CHECK:STDOUT:   %.d32: type = fn_type_with_self_type %As.WithSelf.Convert.type.565, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.189, @T.as_type.as.As.impl.Convert(%ptr.00a, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.a07: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.1a4) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.0c3)]
// CHECK:STDOUT:   %As.impl_witness_table.10e = impl_witness_table (%Core.import_ref.a07), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.47d: @ptr.as.ImplicitAs.impl.992.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.8e0) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.992.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.a18)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c14 = impl_witness_table (%Core.import_ref.47d), @ptr.as.ImplicitAs.impl.992 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.037) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %input.ref: %ptr.037 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.00a]
// CHECK:STDOUT:   %impl.elem0: %.d32 = impl_witness_access constants.%As.impl_witness.cac, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.189]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.00a, constants.%ImplicitAs.facet.9ea) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.00a = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.00a = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.00a = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.00a] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.00a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.00a = wrapper_binding p, %.loc10_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.712 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.712: type = pattern_type %ptr.00a [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.712 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %As.type.ffb: type = facet_type <@As, @As(%ptr.00a)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa489.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.294: %ImplicitAs.type.3aa489.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.1a4: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.294) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.0c3: %T.as_type.as.As.impl.Convert.type.1a4 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c59: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.00a)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.8e0: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.992(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.a18: %ptr.as.ImplicitAs.impl.Convert.type.8e0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.37b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c14, @ptr.as.ImplicitAs.impl.992(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.08f: %ImplicitAs.type.c59 = facet_value %ptr.6b6, (%ImplicitAs.impl_witness.37b) [concrete]
// CHECK:STDOUT:   %As.impl_witness.f61: <witness> = impl_witness imports.%As.impl_witness_table.10e, @T.as_type.as.As.impl(%ptr.00a, %ImplicitAs.facet.08f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.454: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.00a, %ImplicitAs.facet.08f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.8bc: %T.as_type.as.As.impl.Convert.type.454 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.ffb = facet_value %ptr.6b6, (%As.impl_witness.f61) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.5ee: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.00a, %As.facet) [concrete]
// CHECK:STDOUT:   %.b01: type = fn_type_with_self_type %As.WithSelf.Convert.type.5ee, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.8bc, @T.as_type.as.As.impl.Convert(%ptr.00a, %ImplicitAs.facet.08f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a07: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.1a4) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.0c3)]
// CHECK:STDOUT:   %As.impl_witness_table.10e = impl_witness_table (%Core.import_ref.a07), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.47d: @ptr.as.ImplicitAs.impl.992.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.8e0) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.992.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.a18)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c14 = impl_witness_table (%Core.import_ref.47d), @ptr.as.ImplicitAs.impl.992 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.6b6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %input.ref: %ptr.6b6 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.00a]
// CHECK:STDOUT:   %impl.elem0: %.b01 = impl_witness_access constants.%As.impl_witness.f61, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.8bc]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.00a, constants.%ImplicitAs.facet.08f) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.00a = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.00a = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.00a = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.00a] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.00a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.00a = wrapper_binding p, %.loc10_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.712 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.de6: type = pattern_type %ptr.037 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.de6 = value_binding_pattern p [concrete]
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
// CHECK:STDOUT: fn @F(%input.param: %ptr.00a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %input.ref: %ptr.00a = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc13_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc13_38: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc13_40: type = ptr_type %S.ref.loc13_38 [concrete = constants.%ptr.037]
// CHECK:STDOUT:   %.loc13_32: %ptr.037 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc13_22: type = splice_block %ptr.loc13_22 [concrete = constants.%ptr.037] {
// CHECK:STDOUT:     %Cpp.ref.loc13_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc13_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc13_22: type = ptr_type %S.ref.loc13_20 [concrete = constants.%ptr.037]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.037 = wrapper_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.de6 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.fcb = value_binding_pattern p [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.00a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %input.ref: %ptr.00a = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc13_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc13_32: type = ptr_type %C.ref.loc13_31 [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:   %.loc13_28: %ptr.6b6 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc13_18: type = splice_block %ptr.loc13_18 [concrete = constants.%ptr.6b6] {
// CHECK:STDOUT:     %C.ref.loc13_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc13_18: type = ptr_type %C.ref.loc13_17 [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.6b6 = wrapper_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.fcb = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.037: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.de6: type = pattern_type %ptr.037 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.de6 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.ae7: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.037)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.232: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.691: %ptr.as.UnsafeAs.impl.Convert.type.232 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.4ec: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.e4b, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.664: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.cba: %ptr.as.UnsafeAs.impl.Convert.type.664 = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.ae7 = facet_value %ptr.00a, (%UnsafeAs.impl_witness.4ec) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.9c5: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.037, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.d46: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.9c5, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.cba, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.822: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.232) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.691)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.e4b = impl_witness_table (%Core.import_ref.822), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.00a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %input.ref: %ptr.00a = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc10_45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc10_47: type = ptr_type %S.ref.loc10_45 [concrete = constants.%ptr.037]
// CHECK:STDOUT:   %impl.elem0: %.d46 = impl_witness_access constants.%UnsafeAs.impl_witness.4ec, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.cba]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%S) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.037 = call %bound_method.loc10_39.2(%input.ref)
// CHECK:STDOUT:   %.loc10_39.1: %ptr.037 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_39.2: %ptr.037 = converted %input.ref, %.loc10_39.1
// CHECK:STDOUT:   %.loc10_22: type = splice_block %ptr.loc10_22 [concrete = constants.%ptr.037] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc10_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc10_22: type = ptr_type %S.ref.loc10_20 [concrete = constants.%ptr.037]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.037 = wrapper_binding p, %.loc10_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.de6 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.fcb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.1d9: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.6b6)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.232: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.691: %ptr.as.UnsafeAs.impl.Convert.type.232 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.07d: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.e4b, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.bab: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.ec2: %ptr.as.UnsafeAs.impl.Convert.type.bab = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.1d9 = facet_value %ptr.00a, (%UnsafeAs.impl_witness.07d) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.cc9: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.6b6, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.095: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.cc9, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.ec2, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.822: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.232) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.691)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.e4b = impl_witness_table (%Core.import_ref.822), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.00a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %input.ref: %ptr.00a = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc10_38: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc10_39: type = ptr_type %C.ref.loc10_38 [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:   %impl.elem0: %.095 = impl_witness_access constants.%UnsafeAs.impl_witness.07d, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.ec2]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%C) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.6b6 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.6b6 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.6b6 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10_18 [concrete = constants.%ptr.6b6] {
// CHECK:STDOUT:     %C.ref.loc10_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_18: type = ptr_type %C.ref.loc10_17 [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.6b6 = wrapper_binding p, %.loc10_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.fcb = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
