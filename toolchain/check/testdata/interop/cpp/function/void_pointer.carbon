// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/void_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/void_pointer.carbon

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
  let output: Cpp.void* = Cpp.foo();
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
  let output: Core.Optional(Cpp.void*) = Cpp.foo();
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc10_27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %ptr = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_23: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Cpp.ref.loc10_15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_35.1: %ptr = value_of_initializer %foo.call
// CHECK:STDOUT:   %.loc10_35.2: %ptr = converted %foo.call, %.loc10_35.1
// CHECK:STDOUT:   %output: %ptr = value_binding output, %.loc10_35.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nullable_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.8ba: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.4ba: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.d92: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.68f: %ptr.as.OptionalStorage.impl.Some.type.d92 = struct_value () [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.738: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.071, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.b95: %OptionalStorage.type = facet_value %ptr.874, (%OptionalStorage.impl_witness.738) [concrete]
// CHECK:STDOUT:   %Optional.f0f: type = class_type @Optional, @Optional(%OptionalStorage.facet.b95) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.01b: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.f0f)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.7d0: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Optional.f0f) [concrete]
// CHECK:STDOUT:   %OptionalAs.type.08a: type = facet_type <@OptionalAs, @OptionalAs(%T.8ba)> [symbolic]
// CHECK:STDOUT:   %U.729: %OptionalAs.type.08a = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.type.494: type = fn_type @U.binding.as_type.as.ImplicitAs.impl.Convert.2, @U.binding.as_type.as.ImplicitAs.impl.898(%T.8ba, %U.729) [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.858: %U.binding.as_type.as.ImplicitAs.impl.Convert.type.494 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.fb4: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.b95)> [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OptionalAs.impl.Convert.type.526: type = fn_type @T.binding.as_type.as.OptionalAs.impl.Convert, @T.binding.as_type.as.OptionalAs.impl(%T.8ba) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OptionalAs.impl.Convert.613: %T.binding.as_type.as.OptionalAs.impl.Convert.type.526 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.a3f: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.873, @T.binding.as_type.as.OptionalAs.impl(%OptionalStorage.facet.b95) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.fb4 = facet_value %ptr.874, (%OptionalAs.impl_witness.a3f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.732: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9c4, @U.binding.as_type.as.ImplicitAs.impl.898(%OptionalStorage.facet.b95, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.type.3ce: type = fn_type @U.binding.as_type.as.ImplicitAs.impl.Convert.2, @U.binding.as_type.as.ImplicitAs.impl.898(%OptionalStorage.facet.b95, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.564: %U.binding.as_type.as.ImplicitAs.impl.Convert.type.3ce = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.01b = facet_value %ptr.874, (%ImplicitAs.impl_witness.732) [concrete]
// CHECK:STDOUT:   %.0c7: type = fn_type_with_self_type %ImplicitAs.Convert.type.7d0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.binding.as_type.as.ImplicitAs.impl.Convert.564, @U.binding.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.b95, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %facet_value.b55: %type_where = facet_value %Optional.f0f, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.048: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b55) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.ce7: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.048 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = constants.%Cpp.void
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.ecf: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.4ba)]
// CHECK:STDOUT:   %Core.import_ref.902 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c0e: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.d92) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.68f)]
// CHECK:STDOUT:   %Core.import_ref.f15 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.071 = impl_witness_table (%Core.import_ref.ecf, %Core.import_ref.902, %Core.import_ref.c0e, %Core.import_ref.f15, %Core.import_ref.bd4), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_16.1: type = splice_block %Optional [concrete = constants.%Optional.f0f] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.874, (constants.%OptionalStorage.impl_witness.738) [concrete = constants.%OptionalStorage.facet.b95]
// CHECK:STDOUT:       %.loc10_16.2: %OptionalStorage.type = converted constants.%ptr.874, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.b95]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.b95) [concrete = constants.%Optional.f0f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5c5: @U.binding.as_type.as.ImplicitAs.impl.898.%U.binding.as_type.as.ImplicitAs.impl.Convert.type (%U.binding.as_type.as.ImplicitAs.impl.Convert.type.494) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.binding.as_type.as.ImplicitAs.impl.898.%U.binding.as_type.as.ImplicitAs.impl.Convert (constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.858)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9c4 = impl_witness_table (%Core.import_ref.5c5), @U.binding.as_type.as.ImplicitAs.impl.898 [concrete]
// CHECK:STDOUT:   %Core.import_ref.28c: @T.binding.as_type.as.OptionalAs.impl.%T.binding.as_type.as.OptionalAs.impl.Convert.type (%T.binding.as_type.as.OptionalAs.impl.Convert.type.526) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OptionalAs.impl.%T.binding.as_type.as.OptionalAs.impl.Convert (constants.%T.binding.as_type.as.OptionalAs.impl.Convert.613)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.873 = impl_witness_table (%Core.import_ref.28c), @T.binding.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.874) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr.874 = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.0c7 = impl_witness_access constants.%ImplicitAs.impl_witness.732, element0 [concrete = constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.564]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.binding.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.b95, constants.%OptionalAs.facet) [concrete = constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.f0f = call %bound_method.loc10_11.2(%input.ref)
// CHECK:STDOUT:   %.loc10_11.1: init %Optional.f0f = converted %input.ref, %U.binding.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_11.2: ref %Optional.f0f = temporary_storage
// CHECK:STDOUT:   %.loc10_11.3: ref %Optional.f0f = temporary %.loc10_11.2, %.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.4: %Optional.f0f = acquire_value %.loc10_11.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc10_11.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_11.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_11.3(%.loc10_11.3)
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   %T.8ba: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.401: type = fn_type @Optional.None, @Optional(%T.8ba) [symbolic]
// CHECK:STDOUT:   %Optional.None.c2f: %Optional.None.type.401 = struct_value () [symbolic]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %ptr.874: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.4ba: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.241: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.470: %ptr.as.OptionalStorage.impl.None.type.241 = struct_value () [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.1c0: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.4cf, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.874, (%OptionalStorage.impl_witness.1c0) [concrete]
// CHECK:STDOUT:   %Optional.879: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.026: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.ada: %Optional.None.type.026 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.ada, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b51: %type_where = facet_value %Optional.879, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.604: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b51) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.823: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.604 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .CppCompat = %CppCompat.532
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
// CHECK:STDOUT:   %Core.import_ref.559: @Optional.%Optional.None.type (%Optional.None.type.401) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.c2f)]
// CHECK:STDOUT:   %Core.CppCompat: <namespace> = import_ref Core//prelude, CppCompat, loaded
// CHECK:STDOUT:   %CppCompat.532: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .VoidBase = %Core.VoidBase
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.VoidBase: type = import_ref Core//prelude/types/cpp/void, VoidBase, loaded [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %Core.import_ref.ecf: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.4ba)]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.241) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.470)]
// CHECK:STDOUT:   %Core.import_ref.c5b = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f15 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.4cf = impl_witness_table (%Core.import_ref.ecf, %Core.import_ref.13c, %Core.import_ref.c5b, %Core.import_ref.f15, %Core.import_ref.bd4), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_42.1: type = splice_block %Optional [concrete = constants.%Optional.879] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.874, (constants.%OptionalStorage.impl_witness.1c0) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc10_42.2: %OptionalStorage.type = converted constants.%ptr.874, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.879]
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
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.1c0) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc10_34: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.879]
// CHECK:STDOUT:   %.loc10_35: %Optional.None.type.026 = specific_constant imports.%Core.import_ref.559, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.ada]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.026 = name_ref None, %.loc10_35 [concrete = constants.%Optional.None.ada]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.879 = call %Optional.None.specific_fn()
// CHECK:STDOUT:   %.loc10_41.1: ref %Optional.879 = temporary_storage
// CHECK:STDOUT:   %.loc10_41.2: ref %Optional.879 = temporary %.loc10_41.1, %Optional.None.call
// CHECK:STDOUT:   %.loc10_41.3: %Optional.879 = acquire_value %.loc10_41.2
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc10_41.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_41.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.823
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_41.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc10_41.2)
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.4ba: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.47a: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.f51, @ptr.as.OptionalStorage.impl(%Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.874, (%OptionalStorage.impl_witness.47a) [concrete]
// CHECK:STDOUT:   %Optional.add: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.091: type = pattern_type %Optional.add [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.e1e: %type_where = facet_value %Optional.add, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.058: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.e1e) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.edd: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.058 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .CppCompat = %CppCompat.532
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
// CHECK:STDOUT:   %CppCompat.532: <namespace> = namespace %Core.CppCompat, [concrete] {
// CHECK:STDOUT:     .VoidBase = %Core.VoidBase
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.VoidBase: type = import_ref Core//prelude/types/cpp/void, VoidBase, loaded [concrete = constants.%Cpp.void]
// CHECK:STDOUT:   %Core.import_ref.ecf: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.4ba)]
// CHECK:STDOUT:   %Core.import_ref.902 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c5b = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f15 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.f51 = impl_witness_table (%Core.import_ref.ecf, %Core.import_ref.902, %Core.import_ref.c5b, %Core.import_ref.f15, %Core.import_ref.bd4), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.874, (constants.%OptionalStorage.impl_witness.47a) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10: %OptionalStorage.type = converted constants.%ptr.874, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.add]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type.091 = value_binding_pattern output [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %Optional.add = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_38.1: type = splice_block %Optional [concrete = constants.%Optional.add] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc10_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, constants.%Cpp.void [concrete = constants.%Cpp.void]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.874]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.47a) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10_38.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.add]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_50.1: ref %Optional.add = temporary_storage
// CHECK:STDOUT:   %.loc10_50.2: ref %Optional.add = temporary %.loc10_50.1, %foo.call
// CHECK:STDOUT:   %.loc10_50.3: %Optional.add = acquire_value %.loc10_50.2
// CHECK:STDOUT:   %output: %Optional.add = value_binding output, %.loc10_50.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_50.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.edd
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_50.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc10_50.2)
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.e0b, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.12b: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.994: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.12b = struct_value () [concrete]
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %non_nullable_pointer.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.994
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %non_nullable_pointer.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%non_nullable_pointer.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cdc: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.65b: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cdc = struct_value () [concrete]
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
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %const_void_pointer.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.65b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %const_void_pointer.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%const_void_pointer.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
