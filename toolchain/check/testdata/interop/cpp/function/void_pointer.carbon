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

// --- fail_todo_null_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto foo(void* x) -> void;
''';

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_null_param.carbon:[[@LINE+4]]:11: error: call argument of type `Core.Optional(Cpp.void* as Core.OptionalStorage)` is not supported [CppCallArgTypeNotSupported]
  // CHECK:STDERR:   Cpp.foo(Core.Optional(Cpp.void*).None());
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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

// --- fail_void.carbon

library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_void.carbon:[[@LINE+5]]:6: error: parameter has incomplete type `Cpp.void` in function definition [IncompleteTypeInFunctionParam]
// CHECK:STDERR: fn F(x: Cpp.void) {}
// CHECK:STDERR:      ^~~~~~~~~~~
// CHECK:STDERR: fail_void.carbon: note: `Cpp.void` is always-incomplete [CppVoidIncomplete]
// CHECK:STDERR:
fn F(x: Cpp.void) {}

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
// CHECK:STDOUT:   %ptr: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = Cpp.void
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
// CHECK:STDOUT:   %ptr: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = Cpp.void
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
// CHECK:STDOUT:     %void.ref: type = name_ref void, Cpp.void [concrete = Cpp.void]
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
// CHECK:STDOUT:   %ptr.03c: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.3fe: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.4f0: type = ptr_type %T.d9f [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.cff: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.4f0) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.911: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.2a0: %ptr.as.OptionalStorage.impl.Some.type.911 = struct_value () [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.a64: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.236, @ptr.as.OptionalStorage.impl(Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.e2a: %OptionalStorage.type = facet_value %ptr.03c, (%OptionalStorage.impl_witness.a64) [concrete]
// CHECK:STDOUT:   %Optional.75d: type = class_type @Optional, @Optional(%OptionalStorage.facet.e2a) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.707: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.75d)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.465: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Optional.75d) [concrete]
// CHECK:STDOUT:   %OptionalAs.type.593: type = facet_type <@OptionalAs, @OptionalAs(%T.3fe)> [symbolic]
// CHECK:STDOUT:   %U.ec3: %OptionalAs.type.593 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.type.855: type = fn_type @U.binding.as_type.as.ImplicitAs.impl.Convert.2, @U.binding.as_type.as.ImplicitAs.impl.ea7(%T.3fe, %U.ec3) [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.337: %U.binding.as_type.as.ImplicitAs.impl.Convert.type.855 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.a25: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.e2a)> [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OptionalAs.impl.Convert.type.8c6: type = fn_type @T.binding.as_type.as.OptionalAs.impl.Convert, @T.binding.as_type.as.OptionalAs.impl(%T.3fe) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OptionalAs.impl.Convert.180: %T.binding.as_type.as.OptionalAs.impl.Convert.type.8c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.c4d: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.8ee, @T.binding.as_type.as.OptionalAs.impl(%OptionalStorage.facet.e2a) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.a25 = facet_value %ptr.03c, (%OptionalAs.impl_witness.c4d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.354: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.da6, @U.binding.as_type.as.ImplicitAs.impl.ea7(%OptionalStorage.facet.e2a, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.type.efe: type = fn_type @U.binding.as_type.as.ImplicitAs.impl.Convert.2, @U.binding.as_type.as.ImplicitAs.impl.ea7(%OptionalStorage.facet.e2a, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.aa4: %U.binding.as_type.as.ImplicitAs.impl.Convert.type.efe = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.707 = facet_value %ptr.03c, (%ImplicitAs.impl_witness.354) [concrete]
// CHECK:STDOUT:   %.120: type = fn_type_with_self_type %ImplicitAs.Convert.type.465, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.binding.as_type.as.ImplicitAs.impl.Convert.aa4, @U.binding.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.e2a, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %facet_value.9af: %type_where = facet_value %Optional.75d, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.773: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.9af) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.9de: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.773 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.a68: type = ptr_type %Optional.75d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = Cpp.void
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.2fb: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.cff)]
// CHECK:STDOUT:   %Core.import_ref.a7c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.1b2: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.911) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.2a0)]
// CHECK:STDOUT:   %Core.import_ref.6a9 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.971 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.236 = impl_witness_table (%Core.import_ref.2fb, %Core.import_ref.a7c, %Core.import_ref.1b2, %Core.import_ref.6a9, %Core.import_ref.971), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_16.1: type = splice_block %Optional [concrete = constants.%Optional.75d] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.03c, (constants.%OptionalStorage.impl_witness.a64) [concrete = constants.%OptionalStorage.facet.e2a]
// CHECK:STDOUT:       %.loc10_16.2: %OptionalStorage.type = converted constants.%ptr.03c, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.e2a]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.e2a) [concrete = constants.%Optional.75d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ec9: @U.binding.as_type.as.ImplicitAs.impl.ea7.%U.binding.as_type.as.ImplicitAs.impl.Convert.type (%U.binding.as_type.as.ImplicitAs.impl.Convert.type.855) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.binding.as_type.as.ImplicitAs.impl.ea7.%U.binding.as_type.as.ImplicitAs.impl.Convert (constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.337)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.da6 = impl_witness_table (%Core.import_ref.ec9), @U.binding.as_type.as.ImplicitAs.impl.ea7 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7fb: @T.binding.as_type.as.OptionalAs.impl.%T.binding.as_type.as.OptionalAs.impl.Convert.type (%T.binding.as_type.as.OptionalAs.impl.Convert.type.8c6) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OptionalAs.impl.%T.binding.as_type.as.OptionalAs.impl.Convert (constants.%T.binding.as_type.as.OptionalAs.impl.Convert.180)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.8ee = impl_witness_table (%Core.import_ref.7fb), @T.binding.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.03c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr.03c = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.120 = impl_witness_access constants.%ImplicitAs.impl_witness.354, element0 [concrete = constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.aa4]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @U.binding.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.e2a, constants.%OptionalAs.facet) [concrete = constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.75d = call %bound_method.loc10_11.2(%input.ref)
// CHECK:STDOUT:   %.loc10_11.1: init %Optional.75d = converted %input.ref, %U.binding.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_11.2: ref %Optional.75d = temporary_storage
// CHECK:STDOUT:   %.loc10_11.3: ref %Optional.75d = temporary %.loc10_11.2, %.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.4: %Optional.75d = acquire_value %.loc10_11.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc10_11.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_11.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9de
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.a68 = addr_of %.loc10_11.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_11.3(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_null_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Optional.type: type = generic_class_type @Optional [concrete]
// CHECK:STDOUT:   %Optional.generic: %Optional.type = struct_value () [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.3fe: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.193: type = fn_type @Optional.None, @Optional(%T.3fe) [symbolic]
// CHECK:STDOUT:   %Optional.None.dc7: %Optional.None.type.193 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.03c: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.4f0: type = ptr_type %T.d9f [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.cff: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.4f0) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.8ed: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.41a: %ptr.as.OptionalStorage.impl.None.type.8ed = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.328: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.f52, @ptr.as.OptionalStorage.impl(Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.03c, (%OptionalStorage.impl_witness.328) [concrete]
// CHECK:STDOUT:   %Optional.db3: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.623: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.97f: %Optional.None.type.623 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.97f, @Optional.None(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .void = Cpp.void
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.f1d: @Optional.%Optional.None.type (%Optional.None.type.193) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.dc7)]
// CHECK:STDOUT:   %Core.import_ref.2fb: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.cff)]
// CHECK:STDOUT:   %Core.import_ref.1d4: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.8ed) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.41a)]
// CHECK:STDOUT:   %Core.import_ref.720 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6a9 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.971 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.f52 = impl_witness_table (%Core.import_ref.2fb, %Core.import_ref.1d4, %Core.import_ref.720, %Core.import_ref.6a9, %Core.import_ref.971), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Cpp.ref.loc14_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref: type = name_ref void, Cpp.void [concrete = Cpp.void]
// CHECK:STDOUT:   %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.03c]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.328) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc14_34: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.db3]
// CHECK:STDOUT:   %.loc14_35: %Optional.None.type.623 = specific_constant imports.%Core.import_ref.f1d, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.97f]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.623 = name_ref None, %.loc14_35 [concrete = constants.%Optional.None.97f]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %Optional.None.call: init %Optional.db3 = call %Optional.None.specific_fn()
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
// CHECK:STDOUT:   %ptr.03c: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.4f0: type = ptr_type %T.d9f [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.cff: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.4f0) [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.e1a: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.2f8, @ptr.as.OptionalStorage.impl(Cpp.void) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.03c, (%OptionalStorage.impl_witness.e1a) [concrete]
// CHECK:STDOUT:   %Optional.af8: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.f90: type = pattern_type %Optional.af8 [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.2d8: %type_where = facet_value %Optional.af8, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d7f: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.2d8) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.7ff: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d7f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ee2: type = ptr_type %Optional.af8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Optional = %Core.Optional
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = Cpp.void
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.2fb: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.cff)]
// CHECK:STDOUT:   %Core.import_ref.a7c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.720 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6a9 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.971 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.2f8 = impl_witness_table (%Core.import_ref.2fb, %Core.import_ref.a7c, %Core.import_ref.720, %Core.import_ref.6a9, %Core.import_ref.971), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.03c, (constants.%OptionalStorage.impl_witness.e1a) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10: %OptionalStorage.type = converted constants.%ptr.03c, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.af8]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type.f90 = value_binding_pattern output [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.call: init %Optional.af8 = call imports.%foo.decl()
// CHECK:STDOUT:   %.loc10_38.1: type = splice_block %Optional [concrete = constants.%Optional.af8] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc10_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, Cpp.void [concrete = Cpp.void]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.03c]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.e1a) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10_38.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.af8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_50.1: ref %Optional.af8 = temporary_storage
// CHECK:STDOUT:   %.loc10_50.2: ref %Optional.af8 = temporary %.loc10_50.1, %foo.call
// CHECK:STDOUT:   %.loc10_50.3: %Optional.af8 = acquire_value %.loc10_50.2
// CHECK:STDOUT:   %output: %Optional.af8 = value_binding output, %.loc10_50.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_50.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.7ff
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_50.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.ee2 = addr_of %.loc10_50.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.03c: type = ptr_type Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.730: type = ptr_type %ptr.03c [concrete]
// CHECK:STDOUT:   %pattern_type.283: type = pattern_type %ptr.730 [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.type: type = cpp_overload_set_type @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.value: %Return.cpp_overload_set.type = cpp_overload_set_value @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.type: type = fn_type @Return [concrete]
// CHECK:STDOUT:   %Return: %Return.type = struct_value () [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.type: type = cpp_overload_set_type @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.value: %Invoke.cpp_overload_set.type = cpp_overload_set_value @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.type: type = fn_type @Invoke [concrete]
// CHECK:STDOUT:   %Invoke: %Invoke.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.730, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.7d6: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.62a: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.7d6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b8f: type = ptr_type %ptr.730 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = Cpp.void
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
// CHECK:STDOUT:     %non_nullable_pointer.patt: %pattern_type.283 = ref_binding_pattern non_nullable_pointer [concrete]
// CHECK:STDOUT:     %non_nullable_pointer.var_patt: %pattern_type.283 = var_pattern %non_nullable_pointer.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer.var: ref %ptr.730 = var %non_nullable_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return.ref: %Return.cpp_overload_set.type = name_ref Return, imports.%Return.cpp_overload_set.value [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.call: init %ptr.730 = call imports.%Return.decl()
// CHECK:STDOUT:   assign %non_nullable_pointer.var, %Return.call
// CHECK:STDOUT:   %.loc11: type = splice_block %ptr.loc11_38 [concrete = constants.%ptr.730] {
// CHECK:STDOUT:     %Cpp.ref.loc11_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, Cpp.void [concrete = Cpp.void]
// CHECK:STDOUT:     %ptr.loc11_37: type = ptr_type %void.ref [concrete = constants.%ptr.03c]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %ptr.loc11_37 [concrete = constants.%ptr.730]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer: ref %ptr.730 = ref_binding non_nullable_pointer, %non_nullable_pointer.var
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke.ref: %Invoke.cpp_overload_set.type = name_ref Invoke, imports.%Invoke.cpp_overload_set.value [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_pointer.ref: ref %ptr.730 = name_ref non_nullable_pointer, %non_nullable_pointer
// CHECK:STDOUT:   %.loc12: %ptr.730 = acquire_value %non_nullable_pointer.ref
// CHECK:STDOUT:   %Invoke.call: init %empty_tuple.type = call imports.%Invoke.decl(%.loc12)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %non_nullable_pointer.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.62a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %non_nullable_pointer.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.b8f = addr_of %non_nullable_pointer.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %const: type = const_type Cpp.void [concrete]
// CHECK:STDOUT:   %ptr.974: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.39c: type = pattern_type %ptr.974 [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.type: type = cpp_overload_set_type @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.value: %Return.cpp_overload_set.type = cpp_overload_set_value @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.type: type = fn_type @Return [concrete]
// CHECK:STDOUT:   %Return: %Return.type = struct_value () [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.type: type = cpp_overload_set_type @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.value: %Invoke.cpp_overload_set.type = cpp_overload_set_value @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.type: type = fn_type @Invoke [concrete]
// CHECK:STDOUT:   %Invoke: %Invoke.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.974, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.db8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.a82: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.db8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.db4: type = ptr_type %ptr.974 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = Cpp.void
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
// CHECK:STDOUT:     %const_void_pointer.patt: %pattern_type.39c = ref_binding_pattern const_void_pointer [concrete]
// CHECK:STDOUT:     %const_void_pointer.var_patt: %pattern_type.39c = var_pattern %const_void_pointer.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %const_void_pointer.var: ref %ptr.974 = var %const_void_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return.ref: %Return.cpp_overload_set.type = name_ref Return, imports.%Return.cpp_overload_set.value [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.call: init %ptr.974 = call imports.%Return.decl()
// CHECK:STDOUT:   assign %const_void_pointer.var, %Return.call
// CHECK:STDOUT:   %.loc11: type = splice_block %ptr [concrete = constants.%ptr.974] {
// CHECK:STDOUT:     %Cpp.ref.loc11_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, Cpp.void [concrete = Cpp.void]
// CHECK:STDOUT:     %const: type = const_type %void.ref [concrete = constants.%const]
// CHECK:STDOUT:     %ptr: type = ptr_type %const [concrete = constants.%ptr.974]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %const_void_pointer: ref %ptr.974 = ref_binding const_void_pointer, %const_void_pointer.var
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke.ref: %Invoke.cpp_overload_set.type = name_ref Invoke, imports.%Invoke.cpp_overload_set.value [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %const_void_pointer.ref: ref %ptr.974 = name_ref const_void_pointer, %const_void_pointer
// CHECK:STDOUT:   %.loc12: %ptr.974 = acquire_value %const_void_pointer.ref
// CHECK:STDOUT:   %Invoke.call: init %empty_tuple.type = call imports.%Invoke.decl(%.loc12)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %const_void_pointer.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.a82
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %const_void_pointer.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.db4 = addr_of %const_void_pointer.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
