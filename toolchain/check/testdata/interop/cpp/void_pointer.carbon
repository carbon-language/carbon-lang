// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/void_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/void_pointer.carbon

// --- implicit_as_from_cpp_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.S*) {
  //@dump-sem-ir-begin
  let p: Cpp.void* = input;
  //@dump-sem-ir-end
}

// --- implicit_as_from_carbon_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: C*) {
  //@dump-sem-ir-begin
  let p: Cpp.void* = input;
  //@dump-sem-ir-end
}

// --- explicit_as_from_cpp_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.S*) {
  //@dump-sem-ir-begin
  let p: Cpp.void* = input as Cpp.void*;
  //@dump-sem-ir-end
}

// --- explicit_as_from_carbon_class_pointer_to_void_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: C*) {
  //@dump-sem-ir-begin
  let p: Cpp.void* = input as Cpp.void*;
  //@dump-sem-ir-end
}

// --- fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon:[[@LINE+7]]:19: error: cannot convert expression of type `Cpp.void*` to `Cpp.S*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let p: Cpp.S* = input as Cpp.S*;
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon:[[@LINE+4]]:19: note: type `Cpp.void*` does not implement interface `Core.As(Cpp.S*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let p: Cpp.S* = input as Cpp.S*;
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let p: Cpp.S* = input as Cpp.S*;
  //@dump-sem-ir-end
}

// --- fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon:[[@LINE+7]]:15: error: cannot convert expression of type `Cpp.void*` to `C*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let p: C* = input as C*;
  // CHECK:STDERR:               ^~~~~~~~~~~
  // CHECK:STDERR: fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon:[[@LINE+4]]:15: note: type `Cpp.void*` does not implement interface `Core.As(C*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let p: C* = input as C*;
  // CHECK:STDERR:               ^~~~~~~~~~~
  // CHECK:STDERR:
  let p: C* = input as C*;
  //@dump-sem-ir-end
}

// --- unsafe_as_from_void_pointer_to_cpp_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {};
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  let p: Cpp.S* = input unsafe as Cpp.S*;
  //@dump-sem-ir-end
}

// --- unsafe_as_from_void_pointer_to_carbon_class_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {}

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  let p: C* = input unsafe as C*;
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
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.ccd: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.334(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.1ec: %ptr.as.ImplicitAs.impl.Convert.type.ccd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ffd: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.25e, @ptr.as.ImplicitAs.impl.334(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.f61: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.334(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.d8d: %ptr.as.ImplicitAs.impl.Convert.type.f61 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.02d = facet_value %ptr.5c7, (%ImplicitAs.impl_witness.ffd) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b31: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.751: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b31, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.ffe: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.d8d, @ptr.as.ImplicitAs.impl.Convert.1(%S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.654: @ptr.as.ImplicitAs.impl.334.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.ccd) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.334.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.1ec)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.25e = impl_witness_table (%Core.import_ref.654), @ptr.as.ImplicitAs.impl.334 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.5c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.9fb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.5c7 = name_ref input, %input
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10 [concrete = constants.%ptr.874] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.874]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.751 = impl_witness_access constants.%ImplicitAs.impl_witness.ffd, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.d8d]
// CHECK:STDOUT:   %bound_method.loc10_22.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%S) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.ffe]
// CHECK:STDOUT:   %bound_method.loc10_22.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.874 = call %bound_method.loc10_22.2(%input.ref)
// CHECK:STDOUT:   %.loc10_22.1: %ptr.874 = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_22.2: %ptr.874 = converted %input.ref, %.loc10_22.1
// CHECK:STDOUT:   %p: %ptr.874 = value_binding p, %.loc10_22.2
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
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.ccd: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.334(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.1ec: %ptr.as.ImplicitAs.impl.Convert.type.ccd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.39c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.25e, @ptr.as.ImplicitAs.impl.334(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.155: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.334(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.1f8: %ptr.as.ImplicitAs.impl.Convert.type.155 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.02d = facet_value %ptr.31e, (%ImplicitAs.impl_witness.39c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.514: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.9d0: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.514, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.384: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.1f8, @ptr.as.ImplicitAs.impl.Convert.1(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.654: @ptr.as.ImplicitAs.impl.334.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.ccd) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.334.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.1ec)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.25e = impl_witness_table (%Core.import_ref.654), @ptr.as.ImplicitAs.impl.334 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.31e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.9fb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.31e = name_ref input, %input
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10 [concrete = constants.%ptr.874] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.874]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.9d0 = impl_witness_access constants.%ImplicitAs.impl_witness.39c, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.1f8]
// CHECK:STDOUT:   %bound_method.loc10_22.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%C) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.384]
// CHECK:STDOUT:   %bound_method.loc10_22.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.874 = call %bound_method.loc10_22.2(%input.ref)
// CHECK:STDOUT:   %.loc10_22.1: %ptr.874 = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_22.2: %ptr.874 = converted %input.ref, %.loc10_22.1
// CHECK:STDOUT:   %p: %ptr.874 = value_binding p, %.loc10_22.2
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
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.type.6c0: type = fn_type @T.binding.as_type.as.As.impl.Convert, @T.binding.as_type.as.As.impl(%U.67d, %T.738) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.8f1: %T.binding.as_type.as.As.impl.Convert.type.6c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.02d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.874)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.ccd: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.334(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.1ec: %ptr.as.ImplicitAs.impl.Convert.type.ccd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ffd: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.25e, @ptr.as.ImplicitAs.impl.334(%S) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.02d = facet_value %ptr.5c7, (%ImplicitAs.impl_witness.ffd) [concrete]
// CHECK:STDOUT:   %As.impl_witness.9e9: <witness> = impl_witness imports.%As.impl_witness_table, @T.binding.as_type.as.As.impl(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.type.b2a: type = fn_type @T.binding.as_type.as.As.impl.Convert, @T.binding.as_type.as.As.impl(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.e7d: %T.binding.as_type.as.As.impl.Convert.type.b2a = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d24 = facet_value %ptr.5c7, (%As.impl_witness.9e9) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.23e: type = fn_type @As.WithSelf.Convert, @As(%ptr.874, %As.facet) [concrete]
// CHECK:STDOUT:   %.bb8: type = fn_type_with_self_type %As.WithSelf.Convert.type.23e, %As.facet [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.binding.as_type.as.As.impl.Convert.e7d, @T.binding.as_type.as.As.impl.Convert(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.bde: @T.binding.as_type.as.As.impl.%T.binding.as_type.as.As.impl.Convert.type (%T.binding.as_type.as.As.impl.Convert.type.6c0) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.As.impl.%T.binding.as_type.as.As.impl.Convert (constants.%T.binding.as_type.as.As.impl.Convert.8f1)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.bde), @T.binding.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.654: @ptr.as.ImplicitAs.impl.334.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.ccd) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.334.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.1ec)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.25e = impl_witness_table (%Core.import_ref.654), @ptr.as.ImplicitAs.impl.334 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.5c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.9fb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.5c7 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_34: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_39: type = ptr_type %void.ref.loc10_34 [concrete = constants.%ptr.874]
// CHECK:STDOUT:   %impl.elem0: %.bb8 = impl_witness_access constants.%As.impl_witness.9e9, element0 [concrete = constants.%T.binding.as_type.as.As.impl.Convert.e7d]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.binding.as_type.as.As.impl.Convert(constants.%ptr.874, constants.%ImplicitAs.facet) [concrete = constants.%T.binding.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.call: init %ptr.874 = call %bound_method.loc10_28.2(%input.ref)
// CHECK:STDOUT:   %.loc10_28.1: %ptr.874 = value_of_initializer %T.binding.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_28.2: %ptr.874 = converted %input.ref, %.loc10_28.1
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10_18 [concrete = constants.%ptr.874] {
// CHECK:STDOUT:     %Cpp.ref.loc10_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_13: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_18: type = ptr_type %void.ref.loc10_13 [concrete = constants.%ptr.874]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.874 = value_binding p, %.loc10_28.2
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
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.type.6c0: type = fn_type @T.binding.as_type.as.As.impl.Convert, @T.binding.as_type.as.As.impl(%U.67d, %T.738) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.8f1: %T.binding.as_type.as.As.impl.Convert.type.6c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.02d: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.874)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.ccd: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.334(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.1ec: %ptr.as.ImplicitAs.impl.Convert.type.ccd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.39c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.25e, @ptr.as.ImplicitAs.impl.334(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.02d = facet_value %ptr.31e, (%ImplicitAs.impl_witness.39c) [concrete]
// CHECK:STDOUT:   %As.impl_witness.fa1: <witness> = impl_witness imports.%As.impl_witness_table, @T.binding.as_type.as.As.impl(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.type.05b: type = fn_type @T.binding.as_type.as.As.impl.Convert, @T.binding.as_type.as.As.impl(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.483: %T.binding.as_type.as.As.impl.Convert.type.05b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d24 = facet_value %ptr.31e, (%As.impl_witness.fa1) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.056: type = fn_type @As.WithSelf.Convert, @As(%ptr.874, %As.facet) [concrete]
// CHECK:STDOUT:   %.99e: type = fn_type_with_self_type %As.WithSelf.Convert.type.056, %As.facet [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.binding.as_type.as.As.impl.Convert.483, @T.binding.as_type.as.As.impl.Convert(%ptr.874, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bde: @T.binding.as_type.as.As.impl.%T.binding.as_type.as.As.impl.Convert.type (%T.binding.as_type.as.As.impl.Convert.type.6c0) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.As.impl.%T.binding.as_type.as.As.impl.Convert (constants.%T.binding.as_type.as.As.impl.Convert.8f1)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.bde), @T.binding.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.654: @ptr.as.ImplicitAs.impl.334.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.ccd) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.334.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.1ec)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.25e = impl_witness_table (%Core.import_ref.654), @ptr.as.ImplicitAs.impl.334 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.31e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.9fb = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.31e = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_34: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_39: type = ptr_type %void.ref.loc10_34 [concrete = constants.%ptr.874]
// CHECK:STDOUT:   %impl.elem0: %.99e = impl_witness_access constants.%As.impl_witness.fa1, element0 [concrete = constants.%T.binding.as_type.as.As.impl.Convert.483]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.binding.as_type.as.As.impl.Convert(constants.%ptr.874, constants.%ImplicitAs.facet) [concrete = constants.%T.binding.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.binding.as_type.as.As.impl.Convert.call: init %ptr.874 = call %bound_method.loc10_28.2(%input.ref)
// CHECK:STDOUT:   %.loc10_28.1: %ptr.874 = value_of_initializer %T.binding.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_28.2: %ptr.874 = converted %input.ref, %.loc10_28.1
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10_18 [concrete = constants.%ptr.874] {
// CHECK:STDOUT:     %Cpp.ref.loc10_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_13: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_18: type = ptr_type %void.ref.loc10_13 [concrete = constants.%ptr.874]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.874 = value_binding p, %.loc10_28.2
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
// CHECK:STDOUT:   %Cpp.ref.loc17_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc17_31: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc17_33: type = ptr_type %S.ref.loc17_31 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   %.loc17_25: %ptr.5c7 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_15: type = splice_block %ptr.loc17_15 [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc17_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc17_13: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc17_15: type = ptr_type %S.ref.loc17_13 [concrete = constants.%ptr.5c7]
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.506 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc17_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc17_25: type = ptr_type %C.ref.loc17_24 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   %.loc17_21: %ptr.31e = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_11: type = splice_block %ptr.loc17_11 [concrete = constants.%ptr.31e] {
// CHECK:STDOUT:     %C.ref.loc17_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc17_11: type = ptr_type %C.ref.loc17_10 [concrete = constants.%ptr.31e]
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
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.f06: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.2c4: %ptr.as.UnsafeAs.impl.Convert.type.f06 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.080: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.630, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.2b8: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.fe3: %ptr.as.UnsafeAs.impl.Convert.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.0ee = facet_value %ptr.874, (%UnsafeAs.impl_witness.080) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.3f6: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs(%ptr.5c7, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.fba: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.3f6, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.fe3, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.d3b: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.f06) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.2c4)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.630 = impl_witness_table (%Core.import_ref.d3b), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.259 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc10_38: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc10_40: type = ptr_type %S.ref.loc10_38 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   %impl.elem0: %.fba = impl_witness_access constants.%UnsafeAs.impl_witness.080, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.fe3]
// CHECK:STDOUT:   %bound_method.loc10_32.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%S) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_32.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.5c7 = call %bound_method.loc10_32.2(%input.ref)
// CHECK:STDOUT:   %.loc10_32.1: %ptr.5c7 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_32.2: %ptr.5c7 = converted %input.ref, %.loc10_32.1
// CHECK:STDOUT:   %.loc10_15: type = splice_block %ptr.loc10_15 [concrete = constants.%ptr.5c7] {
// CHECK:STDOUT:     %Cpp.ref.loc10_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc10_13: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc10_15: type = ptr_type %S.ref.loc10_13 [concrete = constants.%ptr.5c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.5c7 = value_binding p, %.loc10_32.2
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
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.f06: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.2c4: %ptr.as.UnsafeAs.impl.Convert.type.f06 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.2f5: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.630, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.04a: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.409: %ptr.as.UnsafeAs.impl.Convert.type.04a = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.1ec = facet_value %ptr.874, (%UnsafeAs.impl_witness.2f5) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.d81: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs(%ptr.31e, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.53b: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.d81, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.409, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d3b: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.f06) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.2c4)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.630 = impl_witness_table (%Core.import_ref.d3b), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.506 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc10_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc10_32: type = ptr_type %C.ref.loc10_31 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   %impl.elem0: %.53b = impl_witness_access constants.%UnsafeAs.impl_witness.2f5, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.409]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%C) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.31e = call %bound_method.loc10_28.2(%input.ref)
// CHECK:STDOUT:   %.loc10_28.1: %ptr.31e = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_28.2: %ptr.31e = converted %input.ref, %.loc10_28.1
// CHECK:STDOUT:   %.loc10_11: type = splice_block %ptr.loc10_11 [concrete = constants.%ptr.31e] {
// CHECK:STDOUT:     %C.ref.loc10_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_11: type = ptr_type %C.ref.loc10_10 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.31e = value_binding p, %.loc10_28.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
