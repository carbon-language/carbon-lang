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
// CHECK:STDOUT:   %ptr.055: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.2d0: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.2c3: type = pattern_type %ptr.2d0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.00e: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.2d0)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.77b: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.703(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.fc8: %ptr.as.ImplicitAs.impl.Convert.type.77b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5d0: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5be, @ptr.as.ImplicitAs.impl.703(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.809: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.703(%S) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.bb0: %ptr.as.ImplicitAs.impl.Convert.type.809 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fdc: %ImplicitAs.type.00e = facet_value %ptr.055, (%ImplicitAs.impl_witness.5d0) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a2f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.2d0, %ImplicitAs.facet.fdc) [concrete]
// CHECK:STDOUT:   %.939: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a2f, %ImplicitAs.facet.fdc [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.d09: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.bb0, @ptr.as.ImplicitAs.impl.Convert.1(%S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.3fd: @ptr.as.ImplicitAs.impl.703.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.77b) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.703.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.fc8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5be = impl_witness_table (%Core.import_ref.3fd), @ptr.as.ImplicitAs.impl.703 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.055) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.2c3 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.055 = name_ref input, %input
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.2d0] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.2d0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.939 = impl_witness_access constants.%ImplicitAs.impl_witness.5d0, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.bb0]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%S) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.d09]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.2d0 = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.2d0 = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.2d0 = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %p: %ptr.2d0 = value_binding p, %.loc10_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.6be: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.2d0: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.2c3: type = pattern_type %ptr.2d0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.00e: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.2d0)> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.77b: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.703(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.fc8: %ptr.as.ImplicitAs.impl.Convert.type.77b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7ba: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5be, @ptr.as.ImplicitAs.impl.703(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.b34: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.703(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.513: %ptr.as.ImplicitAs.impl.Convert.type.b34 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e61: %ImplicitAs.type.00e = facet_value %ptr.6be, (%ImplicitAs.impl_witness.7ba) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ebe: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%ptr.2d0, %ImplicitAs.facet.e61) [concrete]
// CHECK:STDOUT:   %.a6f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ebe, %ImplicitAs.facet.e61 [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.specific_fn.794: <specific function> = specific_function %ptr.as.ImplicitAs.impl.Convert.513, @ptr.as.ImplicitAs.impl.Convert.1(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.3fd: @ptr.as.ImplicitAs.impl.703.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.77b) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.703.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.fc8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5be = impl_witness_table (%Core.import_ref.3fd), @ptr.as.ImplicitAs.impl.703 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.6be) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.2c3 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.6be = name_ref input, %input
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10 [concrete = constants.%ptr.2d0] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %void.ref [concrete = constants.%ptr.2d0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.a6f = impl_witness_access constants.%ImplicitAs.impl_witness.7ba, element0 [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.513]
// CHECK:STDOUT:   %bound_method.loc10_29.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.ImplicitAs.impl.Convert.1(constants.%C) [concrete = constants.%ptr.as.ImplicitAs.impl.Convert.specific_fn.794]
// CHECK:STDOUT:   %bound_method.loc10_29.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.call: init %ptr.2d0 = call %bound_method.loc10_29.2(%input.ref)
// CHECK:STDOUT:   %.loc10_29.1: %ptr.2d0 = value_of_initializer %ptr.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_29.2: %ptr.2d0 = converted %input.ref, %.loc10_29.1
// CHECK:STDOUT:   %p: %ptr.2d0 = value_binding p, %.loc10_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_cpp_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.055: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.2d0: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.2c3: type = pattern_type %ptr.2d0 [concrete]
// CHECK:STDOUT:   %As.type.e06: type = facet_type <@As, @As(%ptr.2d0)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6e610f.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.b7e: %ImplicitAs.type.6e610f.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.783: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.b7e) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.4ea: %T.as_type.as.As.impl.Convert.type.783 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.00e: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.2d0)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.77b: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.703(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.fc8: %ptr.as.ImplicitAs.impl.Convert.type.77b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5d0: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5be, @ptr.as.ImplicitAs.impl.703(%S) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fdc: %ImplicitAs.type.00e = facet_value %ptr.055, (%ImplicitAs.impl_witness.5d0) [concrete]
// CHECK:STDOUT:   %As.impl_witness.47c: <witness> = impl_witness imports.%As.impl_witness_table.f5a, @T.as_type.as.As.impl(%ptr.2d0, %ImplicitAs.facet.fdc) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.a0b: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.2d0, %ImplicitAs.facet.fdc) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.3b1: %T.as_type.as.As.impl.Convert.type.a0b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.e06 = facet_value %ptr.055, (%As.impl_witness.47c) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.446: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.2d0, %As.facet) [concrete]
// CHECK:STDOUT:   %.6c9: type = fn_type_with_self_type %As.WithSelf.Convert.type.446, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.3b1, @T.as_type.as.As.impl.Convert(%ptr.2d0, %ImplicitAs.facet.fdc) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.bd1: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.783) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.4ea)]
// CHECK:STDOUT:   %As.impl_witness_table.f5a = impl_witness_table (%Core.import_ref.bd1), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3fd: @ptr.as.ImplicitAs.impl.703.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.77b) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.703.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.fc8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5be = impl_witness_table (%Core.import_ref.3fd), @ptr.as.ImplicitAs.impl.703 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.055) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.2c3 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.055 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.2d0]
// CHECK:STDOUT:   %impl.elem0: %.6c9 = impl_witness_access constants.%As.impl_witness.47c, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.3b1]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.2d0, constants.%ImplicitAs.facet.fdc) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.2d0 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.2d0 = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.2d0 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.2d0] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.2d0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.2d0 = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_as_from_carbon_class_pointer_to_void_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.6be: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.2d0: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type.2c3: type = pattern_type %ptr.2d0 [concrete]
// CHECK:STDOUT:   %As.type.e06: type = facet_type <@As, @As(%ptr.2d0)> [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6e610f.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.67d)> [symbolic]
// CHECK:STDOUT:   %T.b7e: %ImplicitAs.type.6e610f.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.783: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%U.67d, %T.b7e) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.4ea: %T.as_type.as.As.impl.Convert.type.783 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.00e: type = facet_type <@ImplicitAs, @ImplicitAs(%ptr.2d0)> [concrete]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.type.77b: type = fn_type @ptr.as.ImplicitAs.impl.Convert.1, @ptr.as.ImplicitAs.impl.703(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.ImplicitAs.impl.Convert.fc8: %ptr.as.ImplicitAs.impl.Convert.type.77b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7ba: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5be, @ptr.as.ImplicitAs.impl.703(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e61: %ImplicitAs.type.00e = facet_value %ptr.6be, (%ImplicitAs.impl_witness.7ba) [concrete]
// CHECK:STDOUT:   %As.impl_witness.03e: <witness> = impl_witness imports.%As.impl_witness_table.f5a, @T.as_type.as.As.impl(%ptr.2d0, %ImplicitAs.facet.e61) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.type.13a: type = fn_type @T.as_type.as.As.impl.Convert, @T.as_type.as.As.impl(%ptr.2d0, %ImplicitAs.facet.e61) [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.e2b: %T.as_type.as.As.impl.Convert.type.13a = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.e06 = facet_value %ptr.6be, (%As.impl_witness.03e) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.a05: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%ptr.2d0, %As.facet) [concrete]
// CHECK:STDOUT:   %.a73: type = fn_type_with_self_type %As.WithSelf.Convert.type.a05, %As.facet [concrete]
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.specific_fn: <specific function> = specific_function %T.as_type.as.As.impl.Convert.e2b, @T.as_type.as.As.impl.Convert(%ptr.2d0, %ImplicitAs.facet.e61) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bd1: @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert.type (%T.as_type.as.As.impl.Convert.type.783) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.As.impl.%T.as_type.as.As.impl.Convert (constants.%T.as_type.as.As.impl.Convert.4ea)]
// CHECK:STDOUT:   %As.impl_witness_table.f5a = impl_witness_table (%Core.import_ref.bd1), @T.as_type.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3fd: @ptr.as.ImplicitAs.impl.703.%ptr.as.ImplicitAs.impl.Convert.type (%ptr.as.ImplicitAs.impl.Convert.type.77b) = import_ref Core//prelude/types/cpp/void, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.ImplicitAs.impl.703.%ptr.as.ImplicitAs.impl.Convert (constants.%ptr.as.ImplicitAs.impl.Convert.fc8)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5be = impl_witness_table (%Core.import_ref.3fd), @ptr.as.ImplicitAs.impl.703 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.6be) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.2c3 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.6be = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref.loc10_41: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr.loc10_46: type = ptr_type %void.ref.loc10_41 [concrete = constants.%ptr.2d0]
// CHECK:STDOUT:   %impl.elem0: %.a73 = impl_witness_access constants.%As.impl_witness.03e, element0 [concrete = constants.%T.as_type.as.As.impl.Convert.e2b]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.as_type.as.As.impl.Convert(constants.%ptr.2d0, constants.%ImplicitAs.facet.e61) [concrete = constants.%T.as_type.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %T.as_type.as.As.impl.Convert.call: init %ptr.2d0 = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.2d0 = value_of_initializer %T.as_type.as.As.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.2d0 = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_25: type = splice_block %ptr.loc10_25 [concrete = constants.%ptr.2d0] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref.loc10_20: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc10_25: type = ptr_type %void.ref.loc10_20 [concrete = constants.%ptr.2d0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.2d0 = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.2d0: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.055: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %ptr.055 [concrete]
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
// CHECK:STDOUT: fn @F(%input.param: %ptr.2d0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.f1e = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.2d0 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc17_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc17_38: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc17_40: type = ptr_type %S.ref.loc17_38 [concrete = constants.%ptr.055]
// CHECK:STDOUT:   %.loc17_32: %ptr.055 = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_22: type = splice_block %ptr.loc17_22 [concrete = constants.%ptr.055] {
// CHECK:STDOUT:     %Cpp.ref.loc17_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc17_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc17_22: type = ptr_type %S.ref.loc17_20 [concrete = constants.%ptr.055]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.055 = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_explicit_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.2d0: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.6be: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.1c8: type = pattern_type %ptr.6be [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.2d0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.1c8 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.2d0 = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc17_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc17_32: type = ptr_type %C.ref.loc17_31 [concrete = constants.%ptr.6be]
// CHECK:STDOUT:   %.loc17_28: %ptr.6be = converted %input.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc17_18: type = splice_block %ptr.loc17_18 [concrete = constants.%ptr.6be] {
// CHECK:STDOUT:     %C.ref.loc17_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc17_18: type = ptr_type %C.ref.loc17_17 [concrete = constants.%ptr.6be]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.6be = value_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_cpp_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.2d0: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.055: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %ptr.055 [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.93d: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.055)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.ed9: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.83c: %ptr.as.UnsafeAs.impl.Convert.type.ed9 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.887: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.ec8, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.faf: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %S) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.931: %ptr.as.UnsafeAs.impl.Convert.type.faf = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.93d = facet_value %ptr.2d0, (%UnsafeAs.impl_witness.887) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.ac8: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.055, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.ac8, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.931, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %S) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.9c8: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.ed9) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.83c)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.ec8 = impl_witness_table (%Core.import_ref.9c8), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.2d0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.f1e = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.2d0 = name_ref input, %input
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc10_45: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %ptr.loc10_47: type = ptr_type %S.ref.loc10_45 [concrete = constants.%ptr.055]
// CHECK:STDOUT:   %impl.elem0: %.187 = impl_witness_access constants.%UnsafeAs.impl_witness.887, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.931]
// CHECK:STDOUT:   %bound_method.loc10_39.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%S) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_39.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.055 = call %bound_method.loc10_39.2(%input.ref)
// CHECK:STDOUT:   %.loc10_39.1: %ptr.055 = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_39.2: %ptr.055 = converted %input.ref, %.loc10_39.1
// CHECK:STDOUT:   %.loc10_22: type = splice_block %ptr.loc10_22 [concrete = constants.%ptr.055] {
// CHECK:STDOUT:     %Cpp.ref.loc10_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %S.ref.loc10_20: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %ptr.loc10_22: type = ptr_type %S.ref.loc10_20 [concrete = constants.%ptr.055]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.055 = value_binding p, %.loc10_39.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_as_from_void_pointer_to_carbon_class_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.2d0: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.6be: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.1c8: type = pattern_type %ptr.6be [concrete]
// CHECK:STDOUT:   %UnsafeAs.type.556: type = facet_type <@UnsafeAs, @UnsafeAs(%ptr.6be)> [concrete]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.ed9: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%T.67d, %U.091) [symbolic]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.83c: %ptr.as.UnsafeAs.impl.Convert.type.ed9 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnsafeAs.impl_witness.cea: <witness> = impl_witness imports.%UnsafeAs.impl_witness_table.ec8, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.type.810: type = fn_type @ptr.as.UnsafeAs.impl.Convert, @ptr.as.UnsafeAs.impl(%Cpp.void, %C) [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.eea: %ptr.as.UnsafeAs.impl.Convert.type.810 = struct_value () [concrete]
// CHECK:STDOUT:   %UnsafeAs.facet: %UnsafeAs.type.556 = facet_value %ptr.2d0, (%UnsafeAs.impl_witness.cea) [concrete]
// CHECK:STDOUT:   %UnsafeAs.WithSelf.Convert.type.d97: type = fn_type @UnsafeAs.WithSelf.Convert, @UnsafeAs.WithSelf(%ptr.6be, %UnsafeAs.facet) [concrete]
// CHECK:STDOUT:   %.adf: type = fn_type_with_self_type %UnsafeAs.WithSelf.Convert.type.d97, %UnsafeAs.facet [concrete]
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.specific_fn: <specific function> = specific_function %ptr.as.UnsafeAs.impl.Convert.eea, @ptr.as.UnsafeAs.impl.Convert(%Cpp.void, %C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9c8: @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert.type (%ptr.as.UnsafeAs.impl.Convert.type.ed9) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.UnsafeAs.impl.%ptr.as.UnsafeAs.impl.Convert (constants.%ptr.as.UnsafeAs.impl.Convert.83c)]
// CHECK:STDOUT:   %UnsafeAs.impl_witness_table.ec8 = impl_witness_table (%Core.import_ref.9c8), @ptr.as.UnsafeAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.2d0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.1c8 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %input.ref: %ptr.2d0 = name_ref input, %input
// CHECK:STDOUT:   %C.ref.loc10_38: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc10_39: type = ptr_type %C.ref.loc10_38 [concrete = constants.%ptr.6be]
// CHECK:STDOUT:   %impl.elem0: %.adf = impl_witness_access constants.%UnsafeAs.impl_witness.cea, element0 [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.eea]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.UnsafeAs.impl.Convert(constants.%Cpp.void, constants.%C) [concrete = constants.%ptr.as.UnsafeAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.UnsafeAs.impl.Convert.call: init %ptr.6be = call %bound_method.loc10_35.2(%input.ref)
// CHECK:STDOUT:   %.loc10_35.1: %ptr.6be = value_of_initializer %ptr.as.UnsafeAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr.6be = converted %input.ref, %.loc10_35.1
// CHECK:STDOUT:   %.loc10_18: type = splice_block %ptr.loc10_18 [concrete = constants.%ptr.6be] {
// CHECK:STDOUT:     %C.ref.loc10_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_18: type = ptr_type %C.ref.loc10_17 [concrete = constants.%ptr.6be]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr.6be = value_binding p, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
