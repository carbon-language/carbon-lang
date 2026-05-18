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
// CHECK:STDOUT:   %ptr.e03: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.26d: type = pattern_type %ptr.84c [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.74d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.84c)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.949: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.ef7(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.b13: %ptr.as.ImplicitAs.impl.Convert.type.949 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cba, @ptr.as.ImplicitAs.impl.ef7(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.53c: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.ef7(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.fae: %ptr.as.ImplicitAs.impl.Convert.type.53c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.474: %ImplicitAs.type.74d = facet_value %ptr.e03, (%ImplicitAs.impl_witness.0a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.47c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.84c, %ImplicitAs.facet.474) [concrete]
// CHECK:STDOUT:   %.24b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.47c, %ImplicitAs.facet.474 [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.2c6: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.fae, @ptr.as.ImplicitAs.impl.Convert.1(%S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.574: @ptr.as.ImplicitAs.impl.ef7.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.949) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.ef7.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.b13)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cba = impl_witness_table (%Core.import_ref.574), @ptr.as.ImplicitAs.impl.ef7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.e03) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.26d = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.e03 = name_ref input, %input
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.84c] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.84c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.24b = impl_witness_access constants.%ImplicitAs.impl_witness.0a1, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.fae]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%S) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.2c6]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.84c = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.84c = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.84c = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %p: %ptr.84c = value_binding p, %.loc10_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.26d: type = pattern_type %ptr.84c [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.74d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.84c)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.949: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.ef7(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.b13: %ptr.as.ImplicitAs.impl.Convert.type.949 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cba, @ptr.as.ImplicitAs.impl.ef7(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.f4d: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.ef7(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.286: %ptr.as.ImplicitAs.impl.Convert.type.f4d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.01a: %ImplicitAs.type.74d = facet_value %ptr.b04, (%ImplicitAs.impl_witness.4a7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.697: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.84c, %ImplicitAs.facet.01a) [concrete]
// CHECK:STDOUT:   %.51e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.697, %ImplicitAs.facet.01a [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.a7c: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.286, @ptr.as.ImplicitAs.impl.Convert.1(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.574: @ptr.as.ImplicitAs.impl.ef7.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.949) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.ef7.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.b13)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cba = impl_witness_table (%Core.import_ref.574), @ptr.as.ImplicitAs.impl.ef7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.b04) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.26d = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.b04 = name_ref input, %input
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.84c] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.84c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.51e = impl_witness_access constants.%ImplicitAs.impl_witness.4a7, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.286]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%C) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.a7c]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.84c = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.84c = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.84c = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %p: %ptr.84c = value_binding p, %.loc10_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_cpp_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.e03: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.26d: type = pattern_type %ptr.84c [concrete]
// CHECK:STDOUT:   %As.type.419: type = facet_type <@As, @As(%ptr.84c)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.152c1e.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.f66: %ImplicitAs.type.152c1e.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.13a: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.f66) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.2f4: %T.as_type.as.As.impl.Convert.type.13a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.74d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.84c)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.949: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.ef7(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.b13: %ptr.as.ImplicitAs.impl.Convert.type.949 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cba, @ptr.as.ImplicitAs.impl.ef7(%S) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.474: %ImplicitAs.type.74d = facet_value %ptr.e03, (%ImplicitAs.impl_witness.0a1) [concrete]
// CHECK:STDOUT:   %As.impl_witness.008: <witness> = impl_witness imports.%As.impl_witness_table.652, @T.as_type.as.As.impl(%ptr.84c, %ImplicitAs.facet.474) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.f00: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.84c, %ImplicitAs.facet.474) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.5d6: %T.as_type.as.As.impl.Convert.type.f00 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.419 = facet_value %ptr.e03, (%As.impl_witness.008) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.6d8: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.84c, %As.facet) [concrete]
// CHECK:STDOUT:   %.16d: type = fn_type_with_self_type %As.WithSelf.Convert.type.6d8, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.5d6, @T.as_type.as.As.impl.Convert(%ptr.84c, %ImplicitAs.facet.474) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.2a4: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.13a) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.2f4)]
// CHECK:STDOUT:   %As.impl_witness_table.652 = impl_witness_table (%Core.import_ref.2a4), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.574: @ptr.as.ImplicitAs.impl.ef7.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.949) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.ef7.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.b13)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cba = impl_witness_table (%Core.import_ref.574), @ptr.as.ImplicitAs.impl.ef7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.e03) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.26d = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.e03 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.84c]
// CHECK:STDOUT:   %impl.elem0: %.16d = impl_witness_access constants.%As.impl_witness.008, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.5d6]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.84c, constants.%ImplicitAs.facet.474) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.84c = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.84c = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.84c = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.84c] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.84c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.84c = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.26d: type = pattern_type %ptr.84c [concrete]
// CHECK:STDOUT:   %As.type.419: type = facet_type <@As, @As(%ptr.84c)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.152c1e.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.f66: %ImplicitAs.type.152c1e.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.13a: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.f66) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.2f4: %T.as_type.as.As.impl.Convert.type.13a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.74d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.84c)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.949: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.ef7(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.b13: %ptr.as.ImplicitAs.impl.Convert.type.949 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4a7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cba, @ptr.as.ImplicitAs.impl.ef7(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.01a: %ImplicitAs.type.74d = facet_value %ptr.b04, (%ImplicitAs.impl_witness.4a7) [concrete]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.652, @T.as_type.as.As.impl(%ptr.84c, %ImplicitAs.facet.01a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.a5e: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.84c, %ImplicitAs.facet.01a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.b85: %T.as_type.as.As.impl.Convert.type.a5e = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.419 = facet_value %ptr.b04, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.7d1: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.84c, %As.facet) [concrete]
// CHECK:STDOUT:   %.e96: type = fn_type_with_self_type %As.WithSelf.Convert.type.7d1, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.b85, @T.as_type.as.As.impl.Convert(%ptr.84c, %ImplicitAs.facet.01a) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.2a4: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.13a) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.2f4)]
// CHECK:STDOUT:   %As.impl_witness_table.652 = impl_witness_table (%Core.import_ref.2a4), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.574: @ptr.as.ImplicitAs.impl.ef7.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.949) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.ef7.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.b13)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cba = impl_witness_table (%Core.import_ref.574), @ptr.as.ImplicitAs.impl.ef7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.b04) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.26d = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.b04 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.84c]
// CHECK:STDOUT:   %impl.elem0: %.e96 = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.b85]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.84c, constants.%ImplicitAs.facet.01a) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.84c = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.84c = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.84c = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.84c] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.84c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.84c = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.e03: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.f88: type = pattern_type %ptr.e03 [concrete]
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
// CHECK:STDOUT: fn @F(%input.param: %ptr.84c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.f88 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.84c = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc17_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc17_38: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc17_40: type = ptr_type %S.ref.loc17_38 [concrete = constants.%ptr.e03]
// CHECK:STDOUT:   %.loc17_32: %ptr.e03 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_22: type = splice_block %ptr.loc17_22 [concrete = constants.%ptr.e03] {
// CHECK:STDOUT:     %Cpp.ref.loc17_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc17_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc17_22: type = ptr_type %S.ref.loc17_20 [concrete = constants.%ptr.e03]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.e03 = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.23d: type = pattern_type %ptr.b04 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.84c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.23d = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.84c = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc17_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc17_32: type = ptr_type %C.ref.loc17_31 [concrete = constants.%ptr.b04]
// CHECK:STDOUT:   %.loc17_28: %ptr.b04 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_18: type = splice_block %ptr.loc17_18 [concrete = constants.%ptr.b04] {
// CHECK:STDOUT:     %C.ref.loc17_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc17_18: type = ptr_type %C.ref.loc17_17 [concrete = constants.%ptr.b04]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.b04 = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.e03: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.f88: type = pattern_type %ptr.e03 [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.f80: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.e03)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.904: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.7e2: %ptr.as.UnsafeAs.impl.Convert.type.904 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.045: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.d18, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.e23: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.ea2: %ptr.as.UnsafeAs.impl.Convert.type.e23 = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.f80 = facet_value %ptr.84c, (%UnsafeAs.impl_witness.045) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.943: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.e03, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.209: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.943, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.ea2, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.b7c: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.904) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.7e2)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.d18 = impl_witness_table (%Core.import_ref.b7c), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.84c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.f88 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.84c = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc10_45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc10_47: type = ptr_type %S.ref.loc10_45 [concrete = constants.%ptr.e03]
// CHECK:STDOUT:   %impl.elem0: %.209 = impl_witness_access constants.%UnsafeAs.impl_witness.045, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.ea2]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%S) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.e03 = call %bound_method.loc10_39.2(%input.ref)
// CHECK:STDOUT:   %.loc10_39.1: %ptr.e03 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_39.2: %ptr.e03 = converted %input.ref, %.loc10_39.1
// CHECK:STDOUT:   %.loc10_22: type = splice_block %ptr.loc10_22 [concrete = constants.%ptr.e03] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc10_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc10_22: type = ptr_type %S.ref.loc10_20 [concrete = constants.%ptr.e03]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.e03 = value_binding p, %.loc10_39.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.84c: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.b04: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.23d: type = pattern_type %ptr.b04 [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.115: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.b04)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.904: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.7e2: %ptr.as.UnsafeAs.impl.Convert.type.904 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.c1c: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.d18, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.8ee: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.919: %ptr.as.UnsafeAs.impl.Convert.type.8ee = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.115 = facet_value %ptr.84c, (%UnsafeAs.impl_witness.c1c) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.5ad: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.b04, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.78b: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.5ad, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.919, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b7c: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.904) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.7e2)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.d18 = impl_witness_table (%Core.import_ref.b7c), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.84c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.23d = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.84c = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc10_38: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc10_39: type = ptr_type %C.ref.loc10_38 [concrete = constants.%ptr.b04]
// CHECK:STDOUT:   %impl.elem0: %.78b = impl_witness_access constants.%UnsafeAs.impl_witness.c1c, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.919]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%C) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.b04 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.b04 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.b04 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10_18 [concrete = constants.%ptr.b04] {
// CHECK:STDOUT:     %C.ref.loc10_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_18: type = ptr_type %C.ref.loc10_17 [concrete = constants.%ptr.b04]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.b04 = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
