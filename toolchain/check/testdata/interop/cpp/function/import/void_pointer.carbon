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
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.67e: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.6fa: %ptr.as.OptionalStorage.impl.Some.type.67e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d24: %Destroy.type = facet_value %ptr.874, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.28c: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d24) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.7af: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.615, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.207: %OptionalStorage.type = facet_value %ptr.874, (%OptionalStorage.impl_witness.7af) [concrete]
// CHECK:STDOUT:   %Optional.8a2: type = class_type @Optional, @Optional(%OptionalStorage.facet.207) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.839: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.8a2)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.ea5: type = facet_type <@OptionalAs, @OptionalAs(%T.542)> [symbolic]
// CHECK:STDOUT:   %U.2d2: %OptionalAs.type.ea5 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.af3: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.be0(%T.542, %U.2d2) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.726: %U.as_type.as.ImplicitAs.impl.Convert.type.af3 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.94c: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.207)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.427: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.542) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.1c4: %T.as_type.as.OptionalAs.impl.Convert.type.427 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.ef3: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.078, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.207) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.94c = facet_value %ptr.874, (%OptionalAs.impl_witness.ef3) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.722: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.038, @U.as_type.as.ImplicitAs.impl.be0(%OptionalStorage.facet.207, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.a4c: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.be0(%OptionalStorage.facet.207, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.7e8: %U.as_type.as.ImplicitAs.impl.Convert.type.a4c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7a7: %ImplicitAs.type.839 = facet_value %ptr.874, (%ImplicitAs.impl_witness.722) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.faa: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.8a2, %ImplicitAs.facet.7a7) [concrete]
// CHECK:STDOUT:   %.d72: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.faa, %ImplicitAs.facet.7a7 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.7e8, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.207, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc10_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.33a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fb6: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.67e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.6fa)]
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.615 = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.33a, %Core.import_ref.fb6, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_16.1: type = splice_block %Optional [concrete = constants.%Optional.8a2] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.874, (constants.%OptionalStorage.impl_witness.7af) [concrete = constants.%OptionalStorage.facet.207]
// CHECK:STDOUT:       %.loc10_16.2: %OptionalStorage.type = converted constants.%ptr.874, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.207]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.207) [concrete = constants.%Optional.8a2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d41: @U.as_type.as.ImplicitAs.impl.be0.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.af3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.be0.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.726)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.038 = impl_witness_table (%Core.import_ref.d41), @U.as_type.as.ImplicitAs.impl.be0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a9b: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.427) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.1c4)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.078 = impl_witness_table (%Core.import_ref.a9b), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.d72 = impl_witness_access constants.%ImplicitAs.impl_witness.722, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.7e8]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.207, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.8a2 = call %bound_method.loc10_11.2(%input.ref)
// CHECK:STDOUT:   %.loc10_11.1: init %Optional.8a2 = converted %input.ref, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_11.2: ref %Optional.8a2 = temporary_storage
// CHECK:STDOUT:   %.loc10_11.3: ref %Optional.8a2 = temporary %.loc10_11.2, %.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.4: %Optional.8a2 = acquire_value %.loc10_11.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc10_11.4)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc10_11.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc10_11.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_11.1(%self.param: ref %MaybeUnformed.28c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_11.2(%self.param: ref %Optional.8a2) {
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
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.fc5: type = fn_type @Optional.None, @Optional(%T.542) [symbolic]
// CHECK:STDOUT:   %Optional.None.fcb: %Optional.None.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.54e: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.50b: %ptr.as.OptionalStorage.impl.None.type.54e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d24: %Destroy.type = facet_value %ptr.874, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.28c: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d24) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.66b: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.921, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.874, (%OptionalStorage.impl_witness.66b) [concrete]
// CHECK:STDOUT:   %Optional.a97: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.709: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.868: %Optional.None.type.709 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.868, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc10_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .CppCompat = %CppCompat.4b4
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
// CHECK:STDOUT:   %Core.import_ref.6af2: @Optional.%Optional.None.type (%Optional.None.type.fc5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.fcb)]
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.4b4: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .VoidBase = %Core.VoidBase
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.VoidBase: type = import_ref Core//prelude/types/cpp/void, VoidBase, loaded [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %Core.import_ref.6af0: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.c4f: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.54e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.50b)]
// CHECK:STDOUT:   %Core.import_ref.7cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.921 = impl_witness_table (%Core.import_ref.6af0, %Core.import_ref.c4f, %Core.import_ref.7cb, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_42.1: type = splice_block %Optional [concrete = constants.%Optional.a97] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.874, (constants.%OptionalStorage.impl_witness.66b) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc10_42.2: %OptionalStorage.type = converted constants.%ptr.874, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.a97]
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
// CHECK:STDOUT:   %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.874]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.66b) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc10_34: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.a97]
// CHECK:STDOUT:   %.loc10_35: %Optional.None.type.709 = specific_constant imports.%Core.import_ref.6af2, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.868]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.709 = name_ref None, %.loc10_35 [concrete = constants.%Optional.None.868]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.a97 = call %Optional.None.specific_fn()
// CHECK:STDOUT:   %.loc10_41.1: ref %Optional.a97 = temporary_storage
// CHECK:STDOUT:   %.loc10_41.2: ref %Optional.a97 = temporary %.loc10_41.1, %Optional.None.call
// CHECK:STDOUT:   %.loc10_41.3: %Optional.a97 = acquire_value %.loc10_41.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc10_41.3)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc10_41.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc10_41.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_41.1(%self.param: ref %MaybeUnformed.28c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_41.2(%self.param: ref %Optional.a97) {
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
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d24: %Destroy.type = facet_value %ptr.874, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.28c: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d24) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.18e: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.51b, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.874, (%OptionalStorage.impl_witness.18e) [concrete]
// CHECK:STDOUT:   %Optional.9e0: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.0b1: type = pattern_type %Optional.9e0 [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc10_57.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .CppCompat = %CppCompat.4b4
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
// CHECK:STDOUT:   %CppCompat.4b4: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .VoidBase = %Core.VoidBase
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.VoidBase: type = import_ref Core//prelude/types/cpp/void, VoidBase, loaded [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.33a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.7cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.51b = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.33a, %Core.import_ref.7cb, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.874, (constants.%OptionalStorage.impl_witness.18e) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10: %OptionalStorage.type = converted constants.%ptr.874, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.9e0]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type.0b1 = value_binding_pattern output [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10_49: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %Optional.9e0 = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_45.1: type = splice_block %Optional [concrete = constants.%Optional.9e0] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc10_36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.874]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.18e) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10_45.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.9e0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_57.1: ref %Optional.9e0 = temporary_storage
// CHECK:STDOUT:   %.loc10_57.2: ref %Optional.9e0 = temporary %.loc10_57.1, %foo.call
// CHECK:STDOUT:   %.loc10_57.3: %Optional.9e0 = acquire_value %.loc10_57.2
// CHECK:STDOUT:   %output: %Optional.9e0 = value_binding output, %.loc10_57.3
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc10_57.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc10_57.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_57.1(%self.param: ref %MaybeUnformed.28c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_57.2(%self.param: ref %Optional.9e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.e0b: type = ptr_type %ptr.874 [concrete]
// CHECK:STDOUT:   %pattern_type.21b: type = pattern_type %ptr.e0b [concrete]
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
// CHECK:STDOUT:     %non_nullable_pointer.patt: %pattern_type.21b = ref_binding_pattern non_nullable_pointer [concrete]
// CHECK:STDOUT:     %non_nullable_pointer.var_patt: %pattern_type.21b = var_pattern %non_nullable_pointer.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer.var: ref %ptr.e0b = var %non_nullable_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return.ref: %Return.cpp_overload_set.type = name_ref Return, imports.%Return.cpp_overload_set.value [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.call: init %ptr.e0b = call imports.%Return.decl()
// CHECK:STDOUT:   assign %non_nullable_pointer.var, %Return.call
// CHECK:STDOUT:   %.loc11: type = splice_block %ptr.loc11_38 [concrete = constants.%ptr.e0b] {
// CHECK:STDOUT:     %Cpp.ref.loc11_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr.loc11_37: type = ptr_type %void.ref [concrete = constants.%ptr.874]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %ptr.loc11_37 [concrete = constants.%ptr.e0b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer: ref %ptr.e0b = ref_binding non_nullable_pointer, %non_nullable_pointer.var
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke.ref: %Invoke.cpp_overload_set.type = name_ref Invoke, imports.%Invoke.cpp_overload_set.value [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_pointer.ref: ref %ptr.e0b = name_ref non_nullable_pointer, %non_nullable_pointer
// CHECK:STDOUT:   %.loc12: %ptr.e0b = acquire_value %non_nullable_pointer.ref
// CHECK:STDOUT:   %Invoke.call: init %empty_tuple.type = call imports.%Invoke.decl(%.loc12)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %non_nullable_pointer.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%non_nullable_pointer.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %ptr.e0b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %const: type = const_type %Cpp.void [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.972: type = pattern_type %ptr [concrete]
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
// CHECK:STDOUT:     %const_void_pointer.patt: %pattern_type.972 = ref_binding_pattern const_void_pointer [concrete]
// CHECK:STDOUT:     %const_void_pointer.var_patt: %pattern_type.972 = var_pattern %const_void_pointer.patt [concrete]
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
