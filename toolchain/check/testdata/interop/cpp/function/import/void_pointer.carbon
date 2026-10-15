// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/void_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/void_pointer.carbon

// --- non_nullable_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo_non_nullable_param(void* _Nonnull x) -> void;
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  Cpp.foo_non_nullable_param(input);
  //@dump-sem-ir-end
}

// --- non_nullable_return_value.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo_non_nullable_return() -> void* _Nonnull;
''';

fn F() {
  //@dump-sem-ir-begin
  let unused output: Cpp.void* = Cpp.foo_non_nullable_return();
  //@dump-sem-ir-end
}

// --- nullable_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo_nullable_param(void* x) -> void;
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  Cpp.foo_nullable_param(input);
  //@dump-sem-ir-end
}

// --- null_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo_null_param(void* x) -> void;
''';

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo_null_param(Core.Optional(Cpp.void*).None());
  //@dump-sem-ir-end
}

// --- nullable_return_value.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo_nullable_return() -> void*;
''';

fn F() {
  //@dump-sem-ir-begin
  let unused output: Core.Optional(Cpp.void*) = Cpp.foo_nullable_return();
  //@dump-sem-ir-end
}

// --- non_nullable_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto Return_pointer() -> void* _Nonnull* _Nonnull;
auto Invoke_pointer(void* _Nonnull* _Nonnull) -> void;
''';

fn F() {
  //@dump-sem-ir-begin
  var non_nullable_pointer: Cpp.void** = Cpp.Return_pointer();
  Cpp.Invoke_pointer(non_nullable_pointer);
  //@dump-sem-ir-end
}

// --- non_nullable_const.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto Return_const() -> const void* _Nonnull;
auto Invoke_const(const void* _Nonnull) -> void;
''';

fn F() {
  //@dump-sem-ir-begin
  var const_void_pointer: const Cpp.void* = Cpp.Return_const();
  Cpp.Invoke_const(const_void_pointer);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- non_nullable_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %foo_non_nullable_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_non_nullable_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_non_nullable_param.cpp_overload_set.value: %foo_non_nullable_param.cpp_overload_set.type = cpp_overload_set_value @foo_non_nullable_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_non_nullable_param.type: type = fn_type @foo_non_nullable_param [concrete]
// CHECK:STDOUT:   %foo_non_nullable_param: %foo_non_nullable_param.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo_non_nullable_param = %foo_non_nullable_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_non_nullable_param.cpp_overload_set.value: %foo_non_nullable_param.cpp_overload_set.type = cpp_overload_set_value @foo_non_nullable_param.cpp_overload_set [concrete = constants.%foo_non_nullable_param.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_non_nullable_param.decl: %foo_non_nullable_param.type = fn_decl @foo_non_nullable_param [concrete = constants.%foo_non_nullable_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_non_nullable_param.ref: %foo_non_nullable_param.cpp_overload_set.type = name_ref foo_non_nullable_param, imports.%foo_non_nullable_param.cpp_overload_set.value [concrete = constants.%foo_non_nullable_param.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr = name_ref input, %input
// CHECK:STDOUT:   %foo_non_nullable_param.call: init %empty_tuple.type = call imports.%foo_non_nullable_param.decl(%input.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %output.patt: %pattern_type = value_binding_pattern output [concrete]
// CHECK:STDOUT:   %foo_non_nullable_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_non_nullable_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_non_nullable_return.cpp_overload_set.value: %foo_non_nullable_return.cpp_overload_set.type = cpp_overload_set_value @foo_non_nullable_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_non_nullable_return.type: type = fn_type @foo_non_nullable_return [concrete]
// CHECK:STDOUT:   %foo_non_nullable_return: %foo_non_nullable_return.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo_non_nullable_return = %foo_non_nullable_return.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_non_nullable_return.cpp_overload_set.value: %foo_non_nullable_return.cpp_overload_set.type = cpp_overload_set_value @foo_non_nullable_return.cpp_overload_set [concrete = constants.%foo_non_nullable_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_non_nullable_return.decl: %foo_non_nullable_return.type = fn_decl @foo_non_nullable_return [concrete = constants.%foo_non_nullable_return] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10_34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_non_nullable_return.ref: %foo_non_nullable_return.cpp_overload_set.type = name_ref foo_non_nullable_return, imports.%foo_non_nullable_return.cpp_overload_set.value [concrete = constants.%foo_non_nullable_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_non_nullable_return.call: init %ptr = call imports.%foo_non_nullable_return.decl()
// CHECK:STDOUT:   %.loc10_62.1: %ptr = value_of_initializer %foo_non_nullable_return.call
// CHECK:STDOUT:   %.loc10_62.2: %ptr = converted %foo_non_nullable_return.call, %.loc10_62.1
// CHECK:STDOUT:   %.loc10_30: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cpp.ref.loc10_22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %output: %ptr = wrapper_binding output, %.loc10_62.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type = value_binding_pattern output [concrete = constants.%output.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nullable_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %foo_nullable_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_nullable_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_nullable_param.cpp_overload_set.value: %foo_nullable_param.cpp_overload_set.type = cpp_overload_set_value @foo_nullable_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.2d6: %facet_type.8a3 = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.36c: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.2d6) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.b0c: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.bf1: %ptr.as.OptionalStorage.impl.Some.type.b0c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a43: %SubobjectDestroy.type = facet_value %ptr.00a, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e97: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a43) [concrete]
// CHECK:STDOUT:   %facet_value.6a0: %facet_type.8a3 = facet_value %ptr.00a, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.e97) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.91e: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.6a0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3d3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %.a7d: type = maybe_unformed_type %ptr.00a [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.9a2: %OptionalStorage.type = facet_value %ptr.00a, (%OptionalStorage.impl_witness.3d3) [concrete]
// CHECK:STDOUT:   %Optional.7de: type = class_type @Optional, @Optional(%OptionalStorage.facet.9a2) [concrete]
// CHECK:STDOUT:   %foo_nullable_param.type: type = fn_type @foo_nullable_param [concrete]
// CHECK:STDOUT:   %foo_nullable_param: %foo_nullable_param.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.afd: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.7de)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.5bd: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.c66: %OptionalAs.type.5bd = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.c47: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%T.220, %U.c66) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.f83: %U.as_type.as.ImplicitAs.impl.Convert.type.c47 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.2c3: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.9a2)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.b2a: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.c36: %T.as_type.as.OptionalAs.impl.Convert.type.b2a = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.b96: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.98f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.9a2) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.2c3 = facet_value %ptr.00a, (%OptionalAs.impl_witness.b96) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.552: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.653, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.9a2, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.f25: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.9a2, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.f02: %U.as_type.as.ImplicitAs.impl.Convert.type.f25 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b08: %ImplicitAs.type.afd = facet_value %ptr.00a, (%ImplicitAs.impl_witness.552) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.05b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.7de, %ImplicitAs.facet.b08) [concrete]
// CHECK:STDOUT:   %.b42: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.05b, %ImplicitAs.facet.b08 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.f02, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.9a2, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_26.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ff2: %SubobjectDestroy.type = facet_value %Optional.7de, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ce3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ff2) [concrete]
// CHECK:STDOUT:   %facet_value.b6e: %facet_type.8a3 = facet_value %Optional.7de, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.ce3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo_nullable_param = %foo_nullable_param.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_nullable_param.cpp_overload_set.value: %foo_nullable_param.cpp_overload_set.type = cpp_overload_set_value @foo_nullable_param.cpp_overload_set [concrete = constants.%foo_nullable_param.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.f02: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.36c)]
// CHECK:STDOUT:   %Core.import_ref.814 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.470: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.b0c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.bf1)]
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.f02, %Core.import_ref.814, %Core.import_ref.470, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b53: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b53), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %foo_nullable_param.decl: %foo_nullable_param.type = fn_decl @foo_nullable_param [concrete = constants.%foo_nullable_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_31.1: type = splice_block %Optional [concrete = constants.%Optional.7de] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.00a, (constants.%OptionalStorage.impl_witness.3d3) [concrete = constants.%OptionalStorage.facet.9a2]
// CHECK:STDOUT:       %.loc10_31.2: %OptionalStorage.type = converted constants.%ptr.00a, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.9a2]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.9a2) [concrete = constants.%Optional.7de]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.098: @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.c47) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.f83)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.653 = impl_witness_table (%Core.import_ref.098), @U.as_type.as.ImplicitAs.impl.0c0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.49b: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.b2a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.c36)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.98f = impl_witness_table (%Core.import_ref.49b), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.00a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_nullable_param.ref: %foo_nullable_param.cpp_overload_set.type = name_ref foo_nullable_param, imports.%foo_nullable_param.cpp_overload_set.value [concrete = constants.%foo_nullable_param.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr.00a = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.b42 = impl_witness_access constants.%ImplicitAs.impl_witness.552, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.f02]
// CHECK:STDOUT:   %bound_method.loc10_26.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.9a2, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_26.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.7de = call %bound_method.loc10_26.2(%input.ref)
// CHECK:STDOUT:   %.loc10_26.1: init %Optional.7de = converted %input.ref, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_26.2: ref %Optional.7de = temporary_storage
// CHECK:STDOUT:   %.loc10_26.3: ref %Optional.7de = temporary %.loc10_26.2, %.loc10_26.1
// CHECK:STDOUT:   %.loc10_26.4: %Optional.7de = acquire_value %.loc10_26.3
// CHECK:STDOUT:   %foo_nullable_param.call: init %empty_tuple.type = call imports.%foo_nullable_param.decl(%.loc10_26.4)
// CHECK:STDOUT:   %facet_value.loc10_26.1: %facet_type.8a3 = facet_value constants.%Optional.7de, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ce3) [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %.loc10_26.5: %facet_type.8a3 = converted constants.%Optional.7de, %facet_value.loc10_26.1 [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %facet_value.loc10_26.2: %facet_type.8a3 = facet_value constants.%Optional.7de, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ce3) [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %.loc10_26.6: %facet_type.8a3 = converted constants.%Optional.7de, %facet_value.loc10_26.2 [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_26.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_26.1(%self.param: ref %.a7d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_26.2(%self.param: ref %MaybeUnformed.91e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_26.3(%self.param: ref %Optional.7de) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- null_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo_null_param.cpp_overload_set.type: type = cpp_overload_set_type @foo_null_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_null_param.cpp_overload_set.value: %foo_null_param.cpp_overload_set.type = cpp_overload_set_value @foo_null_param.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = parameterized_class_name_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.626: type = fn_type @Optional.None, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.None.3ad: %Optional.None.type.626 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.2d6: %facet_type.8a3 = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.36c: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.2d6) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.629: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.f84: %ptr.as.OptionalStorage.impl.None.type.629 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a43: %SubobjectDestroy.type = facet_value %ptr.00a, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e97: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a43) [concrete]
// CHECK:STDOUT:   %facet_value.6a0: %facet_type.8a3 = facet_value %ptr.00a, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.e97) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.91e: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.6a0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3d3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %.a7d: type = maybe_unformed_type %ptr.00a [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.00a, (%OptionalStorage.impl_witness.3d3) [concrete]
// CHECK:STDOUT:   %Optional.7de: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.f11: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.6db: %Optional.None.type.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.6db, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo_null_param.type: type = fn_type @foo_null_param [concrete]
// CHECK:STDOUT:   %foo_null_param: %foo_null_param.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_52.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ff2: %SubobjectDestroy.type = facet_value %Optional.7de, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ce3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ff2) [concrete]
// CHECK:STDOUT:   %facet_value.b6e: %facet_type.8a3 = facet_value %Optional.7de, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.ce3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .CppCompat = %CppCompat.f0dbf9.1
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_null_param = %foo_null_param.cpp_overload_set.value
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_null_param.cpp_overload_set.value: %foo_null_param.cpp_overload_set.type = cpp_overload_set_value @foo_null_param.cpp_overload_set [concrete = constants.%foo_null_param.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.d6d: @Optional.%Optional.None.type (%Optional.None.type.626) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.3ad)]
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.f0dbf9.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .VoidBase = %Core.VoidBase
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.VoidBase: type = import_ref Core//prelude/types/cpp/void, VoidBase, loaded [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %Core.import_ref.f02: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.36c)]
// CHECK:STDOUT:   %Core.import_ref.9ff: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.629) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.f84)]
// CHECK:STDOUT:   %Core.import_ref.a1a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.f02, %Core.import_ref.9ff, %Core.import_ref.a1a, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %foo_null_param.decl: %foo_null_param.type = fn_decl @foo_null_param [concrete = constants.%foo_null_param] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_53.1: type = splice_block %Optional [concrete = constants.%Optional.7de] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.00a, (constants.%OptionalStorage.impl_witness.3d3) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc10_53.2: %OptionalStorage.type = converted constants.%ptr.00a, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.7de]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_null_param.ref: %foo_null_param.cpp_overload_set.type = name_ref foo_null_param, imports.%foo_null_param.cpp_overload_set.value [concrete = constants.%foo_null_param.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Cpp.ref.loc10_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.00a]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.3d3) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc10_45: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.7de]
// CHECK:STDOUT:   %.loc10_46: %Optional.None.type.f11 = specific_constant imports.%Core.import_ref.d6d, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.6db]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.f11 = name_ref None, %.loc10_46 [concrete = constants.%Optional.None.6db]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.7de = call %Optional.None.specific_fn()
// CHECK:STDOUT:   %.loc10_52.1: ref %Optional.7de = temporary_storage
// CHECK:STDOUT:   %.loc10_52.2: ref %Optional.7de = temporary %.loc10_52.1, %Optional.None.call
// CHECK:STDOUT:   %.loc10_52.3: %Optional.7de = acquire_value %.loc10_52.2
// CHECK:STDOUT:   %foo_null_param.call: init %empty_tuple.type = call imports.%foo_null_param.decl(%.loc10_52.3)
// CHECK:STDOUT:   %facet_value.loc10_52.1: %facet_type.8a3 = facet_value constants.%Optional.7de, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ce3) [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %.loc10_52.4: %facet_type.8a3 = converted constants.%Optional.7de, %facet_value.loc10_52.1 [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %facet_value.loc10_52.2: %facet_type.8a3 = facet_value constants.%Optional.7de, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ce3) [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %.loc10_52.5: %facet_type.8a3 = converted constants.%Optional.7de, %facet_value.loc10_52.2 [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_52.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_52.1(%self.param: ref %.a7d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_52.2(%self.param: ref %MaybeUnformed.91e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_52.3(%self.param: ref %Optional.7de) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- nullable_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Optional.type: type = parameterized_class_name_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.2d6: %facet_type.8a3 = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.36c: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.2d6) [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a43: %SubobjectDestroy.type = facet_value %ptr.00a, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e97: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a43) [concrete]
// CHECK:STDOUT:   %facet_value.6a0: %facet_type.8a3 = facet_value %ptr.00a, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.e97) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.91e: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.6a0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3d3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %.a7d: type = maybe_unformed_type %ptr.00a [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.00a, (%OptionalStorage.impl_witness.3d3) [concrete]
// CHECK:STDOUT:   %Optional.7de: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.332: type = pattern_type %Optional.7de [concrete]
// CHECK:STDOUT:   %output.patt: %pattern_type.332 = value_binding_pattern output [concrete]
// CHECK:STDOUT:   %foo_nullable_return.cpp_overload_set.type: type = cpp_overload_set_type @foo_nullable_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_nullable_return.cpp_overload_set.value: %foo_nullable_return.cpp_overload_set.type = cpp_overload_set_value @foo_nullable_return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_nullable_return.type: type = fn_type @foo_nullable_return [concrete]
// CHECK:STDOUT:   %foo_nullable_return: %foo_nullable_return.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_73.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ff2: %SubobjectDestroy.type = facet_value %Optional.7de, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ce3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ff2) [concrete]
// CHECK:STDOUT:   %facet_value.b6e: %facet_type.8a3 = facet_value %Optional.7de, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.ce3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .CppCompat = %CppCompat.f0dbf9.1
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo_nullable_return = %foo_nullable_return.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.f0dbf9.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .VoidBase = %Core.VoidBase
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.VoidBase: type = import_ref Core//prelude/types/cpp/void, VoidBase, loaded [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %Core.import_ref.f02: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.36c)]
// CHECK:STDOUT:   %Core.import_ref.814 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a1a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.f02, %Core.import_ref.814, %Core.import_ref.a1a, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %foo_nullable_return.cpp_overload_set.value: %foo_nullable_return.cpp_overload_set.type = cpp_overload_set_value @foo_nullable_return.cpp_overload_set [concrete = constants.%foo_nullable_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_nullable_return.decl: %foo_nullable_return.type = fn_decl @foo_nullable_return [concrete = constants.%foo_nullable_return] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.00a, (constants.%OptionalStorage.impl_witness.3d3) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10: %OptionalStorage.type = converted constants.%ptr.00a, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.7de]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10_49: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_nullable_return.ref: %foo_nullable_return.cpp_overload_set.type = name_ref foo_nullable_return, imports.%foo_nullable_return.cpp_overload_set.value [concrete = constants.%foo_nullable_return.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_nullable_return.call: init %Optional.7de = call imports.%foo_nullable_return.decl()
// CHECK:STDOUT:   %.loc10_73.1: ref %Optional.7de = temporary_storage
// CHECK:STDOUT:   %.loc10_73.2: ref %Optional.7de = temporary %.loc10_73.1, %foo_nullable_return.call
// CHECK:STDOUT:   %.loc10_73.3: %Optional.7de = acquire_value %.loc10_73.2
// CHECK:STDOUT:   %.loc10_45.1: type = splice_block %Optional [concrete = constants.%Optional.7de] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc10_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.00a]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.3d3) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10_45.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.7de]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %output: %Optional.7de = wrapper_binding output, %.loc10_73.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type.332 = value_binding_pattern output [concrete = constants.%output.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc10_73.1: %facet_type.8a3 = facet_value constants.%Optional.7de, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ce3) [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %.loc10_73.4: %facet_type.8a3 = converted constants.%Optional.7de, %facet_value.loc10_73.1 [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %facet_value.loc10_73.2: %facet_type.8a3 = facet_value constants.%Optional.7de, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.ce3) [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   %.loc10_73.5: %facet_type.8a3 = converted constants.%Optional.7de, %facet_value.loc10_73.2 [concrete = constants.%facet_value.b6e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_73.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_73.1(%self.param: ref %.a7d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_73.2(%self.param: ref %MaybeUnformed.91e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_73.3(%self.param: ref %Optional.7de) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.00a: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.6f9: type = ptr_type %ptr.00a [concrete]
// CHECK:STDOUT:   %pattern_type.c1e: type = pattern_type %ptr.6f9 [concrete]
// CHECK:STDOUT:   %non_nullable_pointer.patt: %pattern_type.c1e = ref_binding_pattern non_nullable_pointer [concrete]
// CHECK:STDOUT:   %non_nullable_pointer.var_patt: %pattern_type.c1e = var_pattern %non_nullable_pointer.patt [concrete]
// CHECK:STDOUT:   %Return_pointer.cpp_overload_set.type: type = cpp_overload_set_type @Return_pointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return_pointer.cpp_overload_set.value: %Return_pointer.cpp_overload_set.type = cpp_overload_set_value @Return_pointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return_pointer.type: type = fn_type @Return_pointer [concrete]
// CHECK:STDOUT:   %Return_pointer: %Return_pointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Invoke_pointer.cpp_overload_set.type: type = cpp_overload_set_type @Invoke_pointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke_pointer.cpp_overload_set.value: %Invoke_pointer.cpp_overload_set.type = cpp_overload_set_value @Invoke_pointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke_pointer.type: type = fn_type @Invoke_pointer [concrete]
// CHECK:STDOUT:   %Invoke_pointer: %Invoke_pointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.36c: %SubobjectDestroy.type = facet_value %ptr.6f9, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9e0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.36c) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %ptr.6f9, (%DestroyImpl.impl_witness.9e0, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .Return_pointer = %Return_pointer.cpp_overload_set.value
// CHECK:STDOUT:     .Invoke_pointer = %Invoke_pointer.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Return_pointer.cpp_overload_set.value: %Return_pointer.cpp_overload_set.type = cpp_overload_set_value @Return_pointer.cpp_overload_set [concrete = constants.%Return_pointer.cpp_overload_set.value]
// CHECK:STDOUT:   %Return_pointer.decl: %Return_pointer.type = fn_decl @Return_pointer [concrete = constants.%Return_pointer] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Invoke_pointer.cpp_overload_set.value: %Invoke_pointer.cpp_overload_set.type = cpp_overload_set_value @Invoke_pointer.cpp_overload_set [concrete = constants.%Invoke_pointer.cpp_overload_set.value]
// CHECK:STDOUT:   %Invoke_pointer.decl: %Invoke_pointer.type = fn_decl @Invoke_pointer [concrete = constants.%Invoke_pointer] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %non_nullable_pointer.var: ref %ptr.6f9 = var_storage %non_nullable_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return_pointer.ref: %Return_pointer.cpp_overload_set.type = name_ref Return_pointer, imports.%Return_pointer.cpp_overload_set.value [concrete = constants.%Return_pointer.cpp_overload_set.value]
// CHECK:STDOUT:   %Return_pointer.call: init %ptr.6f9 = call imports.%Return_pointer.decl()
// CHECK:STDOUT:   assign %non_nullable_pointer.var, %Return_pointer.call
// CHECK:STDOUT:   %.loc11_38: type = splice_block %ptr.loc11_38 [concrete = constants.%ptr.6f9] {
// CHECK:STDOUT:     %Cpp.ref.loc11_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc11_37: type = ptr_type %void.ref [concrete = constants.%ptr.00a]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %ptr.loc11_37 [concrete = constants.%ptr.6f9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer: ref %ptr.6f9 = wrapper_binding non_nullable_pointer, %non_nullable_pointer.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %non_nullable_pointer.patt: %pattern_type.c1e = ref_binding_pattern non_nullable_pointer [concrete = constants.%non_nullable_pointer.patt]
// CHECK:STDOUT:     %non_nullable_pointer.var_patt: %pattern_type.c1e = var_pattern %non_nullable_pointer.patt [concrete = constants.%non_nullable_pointer.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke_pointer.ref: %Invoke_pointer.cpp_overload_set.type = name_ref Invoke_pointer, imports.%Invoke_pointer.cpp_overload_set.value [concrete = constants.%Invoke_pointer.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_pointer.ref: ref %ptr.6f9 = name_ref non_nullable_pointer, %non_nullable_pointer
// CHECK:STDOUT:   %.loc12: %ptr.6f9 = acquire_value %non_nullable_pointer.ref
// CHECK:STDOUT:   %Invoke_pointer.call: init %empty_tuple.type = call imports.%Invoke_pointer.decl(%.loc12)
// CHECK:STDOUT:   %facet_value.loc11_3.1: %facet_type = facet_value constants.%ptr.6f9, (constants.%DestroyImpl.impl_witness.9e0, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_3.1: %facet_type = converted constants.%ptr.6f9, %facet_value.loc11_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc11_3.2: %facet_type = facet_value constants.%ptr.6f9, (constants.%DestroyImpl.impl_witness.9e0, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_3.2: %facet_type = converted constants.%ptr.6f9, %facet_value.loc11_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%non_nullable_pointer.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %ptr.6f9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %const: type = const_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.c83: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %const_void_pointer.patt: %pattern_type.c83 = ref_binding_pattern const_void_pointer [concrete]
// CHECK:STDOUT:   %const_void_pointer.var_patt: %pattern_type.c83 = var_pattern %const_void_pointer.patt [concrete]
// CHECK:STDOUT:   %Return_const.cpp_overload_set.type: type = cpp_overload_set_type @Return_const.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return_const.cpp_overload_set.value: %Return_const.cpp_overload_set.type = cpp_overload_set_value @Return_const.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return_const.type: type = fn_type @Return_const [concrete]
// CHECK:STDOUT:   %Return_const: %Return_const.type = struct_value () [concrete]
// CHECK:STDOUT:   %Invoke_const.cpp_overload_set.type: type = cpp_overload_set_type @Invoke_const.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke_const.cpp_overload_set.value: %Invoke_const.cpp_overload_set.type = cpp_overload_set_value @Invoke_const.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke_const.type: type = fn_type @Invoke_const [concrete]
// CHECK:STDOUT:   %Invoke_const: %Invoke_const.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a8e: %SubobjectDestroy.type = facet_value %ptr, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c5e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a8e) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %ptr, (%DestroyImpl.impl_witness.c5e, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .Return_const = %Return_const.cpp_overload_set.value
// CHECK:STDOUT:     .Invoke_const = %Invoke_const.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Return_const.cpp_overload_set.value: %Return_const.cpp_overload_set.type = cpp_overload_set_value @Return_const.cpp_overload_set [concrete = constants.%Return_const.cpp_overload_set.value]
// CHECK:STDOUT:   %Return_const.decl: %Return_const.type = fn_decl @Return_const [concrete = constants.%Return_const] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Invoke_const.cpp_overload_set.value: %Invoke_const.cpp_overload_set.type = cpp_overload_set_value @Invoke_const.cpp_overload_set [concrete = constants.%Invoke_const.cpp_overload_set.value]
// CHECK:STDOUT:   %Invoke_const.decl: %Invoke_const.type = fn_decl @Invoke_const [concrete = constants.%Invoke_const] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %const_void_pointer.var: ref %ptr = var_storage %const_void_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return_const.ref: %Return_const.cpp_overload_set.type = name_ref Return_const, imports.%Return_const.cpp_overload_set.value [concrete = constants.%Return_const.cpp_overload_set.value]
// CHECK:STDOUT:   %Return_const.call: init %ptr = call imports.%Return_const.decl()
// CHECK:STDOUT:   assign %const_void_pointer.var, %Return_const.call
// CHECK:STDOUT:   %.loc11_41: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cpp.ref.loc11_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %const: type = const_type %void.ref [concrete = constants.%const]
// CHECK:STDOUT:     %ptr: type = ptr_type %const [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %const_void_pointer: ref %ptr = wrapper_binding const_void_pointer, %const_void_pointer.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %const_void_pointer.patt: %pattern_type.c83 = ref_binding_pattern const_void_pointer [concrete = constants.%const_void_pointer.patt]
// CHECK:STDOUT:     %const_void_pointer.var_patt: %pattern_type.c83 = var_pattern %const_void_pointer.patt [concrete = constants.%const_void_pointer.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke_const.ref: %Invoke_const.cpp_overload_set.type = name_ref Invoke_const, imports.%Invoke_const.cpp_overload_set.value [concrete = constants.%Invoke_const.cpp_overload_set.value]
// CHECK:STDOUT:   %const_void_pointer.ref: ref %ptr = name_ref const_void_pointer, %const_void_pointer
// CHECK:STDOUT:   %.loc12: %ptr = acquire_value %const_void_pointer.ref
// CHECK:STDOUT:   %Invoke_const.call: init %empty_tuple.type = call imports.%Invoke_const.decl(%.loc12)
// CHECK:STDOUT:   %facet_value.loc11_3.1: %facet_type = facet_value constants.%ptr, (constants.%DestroyImpl.impl_witness.c5e, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_3.1: %facet_type = converted constants.%ptr, %facet_value.loc11_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc11_3.2: %facet_type = facet_value constants.%ptr, (constants.%DestroyImpl.impl_witness.c5e, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_3.2: %facet_type = converted constants.%ptr, %facet_value.loc11_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%const_void_pointer.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
