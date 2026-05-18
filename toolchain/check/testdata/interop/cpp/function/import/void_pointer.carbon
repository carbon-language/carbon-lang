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
auto foo(void* _Nonnull x) -> void;
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  Cpp.foo(input);
  //@dump-sem-ir-end
}

// --- non_nullable_return_value.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo() -> void* _Nonnull;
''';

fn F() {
  //@dump-sem-ir-begin
  let unused output: Cpp.void* = Cpp.foo();
  //@dump-sem-ir-end
}

// --- nullable_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo(void* x) -> void;
''';

fn F(input: Cpp.void*) {
  //@dump-sem-ir-begin
  Cpp.foo(input);
  //@dump-sem-ir-end
}

// --- null_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo(void* x) -> void;
''';

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(Core.Optional(Cpp.void*).None());
  //@dump-sem-ir-end
}

// --- nullable_return_value.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo() -> void*;
''';

fn F() {
  //@dump-sem-ir-begin
  let unused output: Core.Optional(Cpp.void*) = Cpp.foo();
  //@dump-sem-ir-end
}

// --- non_nullable_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto Return() -> void* _Nonnull* _Nonnull;
auto Invoke(void* _Nonnull* _Nonnull) -> void;
''';

fn F() {
  //@dump-sem-ir-begin
  var non_nullable_pointer: Cpp.void** = Cpp.Return();
  Cpp.Invoke(non_nullable_pointer);
  //@dump-sem-ir-end
}

// --- non_nullable_const.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto Return() -> const void* _Nonnull;
auto Invoke(const void* _Nonnull) -> void;
''';

fn F() {
  //@dump-sem-ir-begin
  var const_void_pointer: const Cpp.void* = Cpp.Return();
  Cpp.Invoke(const_void_pointer);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- non_nullable_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr = name_ref input, %input
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%input.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type = value_binding_pattern output [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10_34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_30: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cpp.ref.loc10_22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_42.1: %ptr = value_of_initializer %foo.call
// CHECK:STDOUT:   %.loc10_42.2: %ptr = converted %foo.call, %.loc10_42.1
// CHECK:STDOUT:   %output: %ptr = value_binding output, %.loc10_42.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nullable_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.209: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.390: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.310: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0b6: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.310) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.4fe: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.0b6) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.876: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.0f5: %ptr.as.OptionalStorage.impl.Some.type.876 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.1: %Destroy.Op.type.60afa5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.275260.1: <witness> = custom_witness (%Destroy.Op.b63e49.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.62b: %Destroy.type = facet_value %ptr.209, (%custom_witness.275260.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.683: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.62b) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.473: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.d60, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.d42: %OptionalStorage.type = facet_value %ptr.209, (%OptionalStorage.impl_witness.473) [concrete]
// CHECK:STDOUT:   %Optional.01d: type = class_type @Optional, @Optional(%OptionalStorage.facet.d42) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.83e: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.01d)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.c8f: type = facet_type <@OptionalAs, @OptionalAs(%T.390)> [symbolic]
// CHECK:STDOUT:   %U.00a: %OptionalAs.type.c8f = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ba3: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.20a(%T.390, %U.00a) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.7c5: %U.as_type.as.ImplicitAs.impl.Convert.type.ba3 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.7bb: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.d42)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.135: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.390) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.8c2: %T.as_type.as.OptionalAs.impl.Convert.type.135 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.a7d: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.d46, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.d42) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.7bb = facet_value %ptr.209, (%OptionalAs.impl_witness.a7d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d50: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d00, @U.as_type.as.ImplicitAs.impl.20a(%OptionalStorage.facet.d42, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.2df: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.20a(%OptionalStorage.facet.d42, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.7e4: %U.as_type.as.ImplicitAs.impl.Convert.type.2df = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1cd: %ImplicitAs.type.83e = facet_value %ptr.209, (%ImplicitAs.impl_witness.d50) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.565: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.01d, %ImplicitAs.facet.1cd) [concrete]
// CHECK:STDOUT:   %.68d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.565, %ImplicitAs.facet.1cd [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.7e4, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.d42, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.3: type = fn_type @Destroy.Op.loc10_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.3: %Destroy.Op.type.60afa5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.7ae: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.4fe)]
// CHECK:STDOUT:   %Core.import_ref.efd = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6a0: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.876) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.0f5)]
// CHECK:STDOUT:   %Core.import_ref.4ad = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.291 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.d60 = impl_witness_table (%Core.import_ref.7ae, %Core.import_ref.efd, %Core.import_ref.6a0, %Core.import_ref.4ad, %Core.import_ref.291), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_16.1: type = splice_block %Optional [concrete = constants.%Optional.01d] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.209, (constants.%OptionalStorage.impl_witness.473) [concrete = constants.%OptionalStorage.facet.d42]
// CHECK:STDOUT:       %.loc10_16.2: %OptionalStorage.type = converted constants.%ptr.209, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.d42]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.d42) [concrete = constants.%Optional.01d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.0bb: @U.as_type.as.ImplicitAs.impl.20a.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ba3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.20a.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.7c5)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d00 = impl_witness_table (%Core.import_ref.0bb), @U.as_type.as.ImplicitAs.impl.20a [concrete]
// CHECK:STDOUT:   %Core.import_ref.1a5: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.135) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.8c2)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.d46 = impl_witness_table (%Core.import_ref.1a5), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.209) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr.209 = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.68d = impl_witness_access constants.%ImplicitAs.impl_witness.d50, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.7e4]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.d42, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.01d = call %bound_method.loc10_11.2(%input.ref)
// CHECK:STDOUT:   %.loc10_11.1: init %Optional.01d = converted %input.ref, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_11.2: ref %Optional.01d = temporary_storage
// CHECK:STDOUT:   %.loc10_11.3: ref %Optional.01d = temporary %.loc10_11.2, %.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.4: %Optional.01d = acquire_value %.loc10_11.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc10_11.4)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc10_11.3, constants.%Destroy.Op.b63e49.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc10_11.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_11.1(%self.param: ref %MaybeUnformed.683) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_11.2(%self.param: ref %Optional.01d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- null_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.390: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.e3d: type = fn_type @Optional.None, @Optional(%T.390) [symbolic]
// CHECK:STDOUT:   %Optional.None.4c5: %Optional.None.type.e3d = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.209: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.310: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0b6: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.310) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.4fe: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.0b6) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.ed5: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.503: %ptr.as.OptionalStorage.impl.None.type.ed5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.1: %Destroy.Op.type.60afa5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.275260.1: <witness> = custom_witness (%Destroy.Op.b63e49.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.62b: %Destroy.type = facet_value %ptr.209, (%custom_witness.275260.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.683: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.62b) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.e16: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.698, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.209, (%OptionalStorage.impl_witness.e16) [concrete]
// CHECK:STDOUT:   %Optional.c0c: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.7c2: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.14f: %Optional.None.type.7c2 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.14f, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.3: type = fn_type @Destroy.Op.loc10_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.3: %Destroy.Op.type.60afa5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .CppCompat = %CppCompat.bd5028.1
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.b28: @Optional.%Optional.None.type (%Optional.None.type.e3d) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.4c5)]
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.bd5028.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .VoidBase = %Core.VoidBase
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.VoidBase: type = import_ref Core//prelude/types/cpp/void, VoidBase, loaded [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %Core.import_ref.7ae: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.4fe)]
// CHECK:STDOUT:   %Core.import_ref.f1b: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.ed5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.503)]
// CHECK:STDOUT:   %Core.import_ref.5aa = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.4ad = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.291 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.698 = impl_witness_table (%Core.import_ref.7ae, %Core.import_ref.f1b, %Core.import_ref.5aa, %Core.import_ref.4ad, %Core.import_ref.291), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_42.1: type = splice_block %Optional [concrete = constants.%Optional.c0c] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.209, (constants.%OptionalStorage.impl_witness.e16) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc10_42.2: %OptionalStorage.type = converted constants.%ptr.209, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.c0c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Cpp.ref.loc10_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.209]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.e16) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc10_34: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.c0c]
// CHECK:STDOUT:   %.loc10_35: %Optional.None.type.7c2 = specific_constant imports.%Core.import_ref.b28, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.14f]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.7c2 = name_ref None, %.loc10_35 [concrete = constants.%Optional.None.14f]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.c0c = call %Optional.None.specific_fn()
// CHECK:STDOUT:   %.loc10_41.1: ref %Optional.c0c = temporary_storage
// CHECK:STDOUT:   %.loc10_41.2: ref %Optional.c0c = temporary %.loc10_41.1, %Optional.None.call
// CHECK:STDOUT:   %.loc10_41.3: %Optional.c0c = acquire_value %.loc10_41.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc10_41.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc10_41.2, constants.%Destroy.Op.b63e49.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc10_41.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_41.1(%self.param: ref %MaybeUnformed.683) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_41.2(%self.param: ref %Optional.c0c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nullable_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.209: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.310: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0b6: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.310) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.4fe: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.0b6) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.1: %Destroy.Op.type.60afa5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.275260.1: <witness> = custom_witness (%Destroy.Op.b63e49.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.62b: %Destroy.type = facet_value %ptr.209, (%custom_witness.275260.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.683: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.62b) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.b10: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.b94, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.209, (%OptionalStorage.impl_witness.b10) [concrete]
// CHECK:STDOUT:   %Optional.ac0: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.ecf: type = pattern_type %Optional.ac0 [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.3: type = fn_type @Destroy.Op.loc10_57.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.3: %Destroy.Op.type.60afa5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .CppCompat = %CppCompat.bd5028.1
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.bd5028.1: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .VoidBase = %Core.VoidBase
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.VoidBase: type = import_ref Core//prelude/types/cpp/void, VoidBase, loaded [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %Core.import_ref.7ae: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.4fe)]
// CHECK:STDOUT:   %Core.import_ref.efd = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5aa = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.4ad = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.291 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.b94 = impl_witness_table (%Core.import_ref.7ae, %Core.import_ref.efd, %Core.import_ref.5aa, %Core.import_ref.4ad, %Core.import_ref.291), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.209, (constants.%OptionalStorage.impl_witness.b10) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10: %OptionalStorage.type = converted constants.%ptr.209, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.ac0]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type.ecf = value_binding_pattern output [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10_49: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %Optional.ac0 = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_45.1: type = splice_block %Optional [concrete = constants.%Optional.ac0] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc10_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.209]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.b10) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10_45.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.ac0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_57.1: ref %Optional.ac0 = temporary_storage
// CHECK:STDOUT:   %.loc10_57.2: ref %Optional.ac0 = temporary %.loc10_57.1, %foo.call
// CHECK:STDOUT:   %.loc10_57.3: %Optional.ac0 = acquire_value %.loc10_57.2
// CHECK:STDOUT:   %output: %Optional.ac0 = value_binding output, %.loc10_57.3
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc10_57.2, constants.%Destroy.Op.b63e49.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc10_57.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_57.1(%self.param: ref %MaybeUnformed.683) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_57.2(%self.param: ref %Optional.ac0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.209: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.fd8: type = ptr_type %ptr.209 [concrete]
// CHECK:STDOUT:   %pattern_type.5c3: type = pattern_type %ptr.fd8 [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.type: type = cpp_overload_set_type @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.value: %Return.cpp_overload_set.type = cpp_overload_set_value @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.type: type = fn_type @Return [concrete]
// CHECK:STDOUT:   %Return: %Return.type = struct_value () [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.type: type = cpp_overload_set_type @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.value: %Invoke.cpp_overload_set.type = cpp_overload_set_value @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.type: type = fn_type @Invoke [concrete]
// CHECK:STDOUT:   %Invoke: %Invoke.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .Return = %Return.cpp_overload_set.value
// CHECK:STDOUT:     .Invoke = %Invoke.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Return.cpp_overload_set.value: %Return.cpp_overload_set.type = cpp_overload_set_value @Return.cpp_overload_set [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.decl: %Return.type = fn_decl @Return [concrete = constants.%Return] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Invoke.cpp_overload_set.value: %Invoke.cpp_overload_set.type = cpp_overload_set_value @Invoke.cpp_overload_set [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %Invoke.decl: %Invoke.type = fn_decl @Invoke [concrete = constants.%Invoke] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %non_nullable_pointer.patt: %pattern_type.5c3 = ref_binding_pattern non_nullable_pointer [concrete]
// CHECK:STDOUT:     %non_nullable_pointer.var_patt: %pattern_type.5c3 = var_pattern %non_nullable_pointer.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer.var: ref %ptr.fd8 = var %non_nullable_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return.ref: %Return.cpp_overload_set.type = name_ref Return, imports.%Return.cpp_overload_set.value [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.call: init %ptr.fd8 = call imports.%Return.decl()
// CHECK:STDOUT:   assign %non_nullable_pointer.var, %Return.call
// CHECK:STDOUT:   %.loc11: type = splice_block %ptr.loc11_38 [concrete = constants.%ptr.fd8] {
// CHECK:STDOUT:     %Cpp.ref.loc11_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc11_37: type = ptr_type %void.ref [concrete = constants.%ptr.209]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %ptr.loc11_37 [concrete = constants.%ptr.fd8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer: ref %ptr.fd8 = ref_binding non_nullable_pointer, %non_nullable_pointer.var
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke.ref: %Invoke.cpp_overload_set.type = name_ref Invoke, imports.%Invoke.cpp_overload_set.value [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_pointer.ref: ref %ptr.fd8 = name_ref non_nullable_pointer, %non_nullable_pointer
// CHECK:STDOUT:   %.loc12: %ptr.fd8 = acquire_value %non_nullable_pointer.ref
// CHECK:STDOUT:   %Invoke.call: init %empty_tuple.type = call imports.%Invoke.decl(%.loc12)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %non_nullable_pointer.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%non_nullable_pointer.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr.fd8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %const: type = const_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.b42: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.type: type = cpp_overload_set_type @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.value: %Return.cpp_overload_set.type = cpp_overload_set_value @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.type: type = fn_type @Return [concrete]
// CHECK:STDOUT:   %Return: %Return.type = struct_value () [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.type: type = cpp_overload_set_type @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.value: %Invoke.cpp_overload_set.type = cpp_overload_set_value @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.type: type = fn_type @Invoke [concrete]
// CHECK:STDOUT:   %Invoke: %Invoke.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .Return = %Return.cpp_overload_set.value
// CHECK:STDOUT:     .Invoke = %Invoke.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Return.cpp_overload_set.value: %Return.cpp_overload_set.type = cpp_overload_set_value @Return.cpp_overload_set [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.decl: %Return.type = fn_decl @Return [concrete = constants.%Return] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Invoke.cpp_overload_set.value: %Invoke.cpp_overload_set.type = cpp_overload_set_value @Invoke.cpp_overload_set [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %Invoke.decl: %Invoke.type = fn_decl @Invoke [concrete = constants.%Invoke] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %const_void_pointer.patt: %pattern_type.b42 = ref_binding_pattern const_void_pointer [concrete]
// CHECK:STDOUT:     %const_void_pointer.var_patt: %pattern_type.b42 = var_pattern %const_void_pointer.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %const_void_pointer.var: ref %ptr = var %const_void_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return.ref: %Return.cpp_overload_set.type = name_ref Return, imports.%Return.cpp_overload_set.value [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.call: init %ptr = call imports.%Return.decl()
// CHECK:STDOUT:   assign %const_void_pointer.var, %Return.call
// CHECK:STDOUT:   %.loc11: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cpp.ref.loc11_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %const: type = const_type %void.ref [concrete = constants.%const]
// CHECK:STDOUT:     %ptr: type = ptr_type %const [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %const_void_pointer: ref %ptr = ref_binding const_void_pointer, %const_void_pointer.var
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke.ref: %Invoke.cpp_overload_set.type = name_ref Invoke, imports.%Invoke.cpp_overload_set.value [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %const_void_pointer.ref: ref %ptr = name_ref const_void_pointer, %const_void_pointer
// CHECK:STDOUT:   %.loc12: %ptr = acquire_value %const_void_pointer.ref
// CHECK:STDOUT:   %Invoke.call: init %empty_tuple.type = call imports.%Invoke.decl(%.loc12)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %const_void_pointer.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%const_void_pointer.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr) = "no_op";
// CHECK:STDOUT:
