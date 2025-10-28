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

// --- todo_fail_void.carbon

library "[[@TEST_NAME]]";

import Cpp inline "";

fn F(x: Cpp.void);

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
// CHECK:STDOUT:   %ptr: type = ptr_type cpp_void_type [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = cpp_void_type
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
// CHECK:STDOUT:   %ptr: type = ptr_type cpp_void_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = cpp_void_type
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
// CHECK:STDOUT:     %void.ref: type = name_ref void, cpp_void_type [concrete = cpp_void_type]
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
// CHECK:STDOUT:   %ptr.5da: type = ptr_type cpp_void_type [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.76d: %OptionalStorage.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.94c: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.4fc: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.566: %ptr.as.OptionalStorage.impl.Some.type.4fc = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.94c: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.377, @ptr.as.OptionalStorage.impl(cpp_void_type) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.5da, (%OptionalStorage.impl_witness.94c) [concrete]
// CHECK:STDOUT:   %Optional.04e: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.710: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.04e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.91f: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Optional.04e) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert.1, @T.binding.as_type.as.ImplicitAs.impl.3a5(%T.76d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.6ac: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5fe: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.7bf, @T.binding.as_type.as.ImplicitAs.impl.3a5(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.7b7: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert.1, @T.binding.as_type.as.ImplicitAs.impl.3a5(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.00c: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.7b7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.710 = facet_value %ptr.5da, (%ImplicitAs.impl_witness.5fe) [concrete]
// CHECK:STDOUT:   %.68d: type = fn_type_with_self_type %ImplicitAs.Convert.type.91f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %T.binding.as_type.as.ImplicitAs.impl.Convert.00c, @T.binding.as_type.as.ImplicitAs.impl.Convert.1(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Optional.04e, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a2c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.654: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.a2c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c4c: type = ptr_type %Optional.04e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = cpp_void_type
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.8c0: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.94c)]
// CHECK:STDOUT:   %Core.import_ref.566 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.637: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.4fc) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.566)]
// CHECK:STDOUT:   %Core.import_ref.6db = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5a7 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.377 = impl_witness_table (%Core.import_ref.8c0, %Core.import_ref.566, %Core.import_ref.637, %Core.import_ref.6db, %Core.import_ref.5a7), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_16.1: type = splice_block %Optional [concrete = constants.%Optional.04e] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5da, (constants.%OptionalStorage.impl_witness.94c) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc10_16.2: %OptionalStorage.type = converted constants.%ptr.5da, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.04e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e04: @T.binding.as_type.as.ImplicitAs.impl.3a5.%T.binding.as_type.as.ImplicitAs.impl.Convert.type (%T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.ImplicitAs.impl.3a5.%T.binding.as_type.as.ImplicitAs.impl.Convert (constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.6ac)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.7bf = impl_witness_table (%Core.import_ref.e04), @T.binding.as_type.as.ImplicitAs.impl.3a5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.5da) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr.5da = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.68d = impl_witness_access constants.%ImplicitAs.impl_witness.5fe, element0 [concrete = constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.00c]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.binding.as_type.as.ImplicitAs.impl.Convert.1(constants.%OptionalStorage.facet) [concrete = constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %.loc10_11.1: ref %Optional.04e = temporary_storage
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.04e = call %bound_method.loc10_11.2(%input.ref) to %.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.2: init %Optional.04e = converted %input.ref, %T.binding.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_11.3: ref %Optional.04e = temporary %.loc10_11.1, %.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.4: %Optional.04e = bind_value %.loc10_11.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc10_11.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_11.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.654
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.c4c = addr_of %.loc10_11.3
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
// CHECK:STDOUT:   %T.76d: %OptionalStorage.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.None.type.3a2: type = fn_type @Optional.None, @Optional(%T.76d) [symbolic]
// CHECK:STDOUT:   %Optional.None.96b: %Optional.None.type.3a2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.5da: type = ptr_type cpp_void_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.94c: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.e66: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.b1f: %ptr.as.OptionalStorage.impl.None.type.e66 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.dd8: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.534, @ptr.as.OptionalStorage.impl(cpp_void_type) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.5da, (%OptionalStorage.impl_witness.dd8) [concrete]
// CHECK:STDOUT:   %Optional.7ef: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.280: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.1da: %Optional.None.type.280 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.1da, @Optional.None(%OptionalStorage.facet) [concrete]
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
// CHECK:STDOUT:     .void = cpp_void_type
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.a6c: @Optional.%Optional.None.type (%Optional.None.type.3a2) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.None (constants.%Optional.None.96b)]
// CHECK:STDOUT:   %Core.import_ref.8c0: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.94c)]
// CHECK:STDOUT:   %Core.import_ref.619: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.e66) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.b1f)]
// CHECK:STDOUT:   %Core.import_ref.cd4 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6db = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5a7 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.534 = impl_witness_table (%Core.import_ref.8c0, %Core.import_ref.619, %Core.import_ref.cd4, %Core.import_ref.6db, %Core.import_ref.5a7), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Cpp.ref.loc14_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %void.ref: type = name_ref void, cpp_void_type [concrete = cpp_void_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.5da]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.dd8) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc14_34: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.7ef]
// CHECK:STDOUT:   %.loc14_35: %Optional.None.type.280 = specific_constant imports.%Core.import_ref.a6c, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.1da]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.280 = name_ref None, %.loc14_35 [concrete = constants.%Optional.None.1da]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %.loc14_41: ref %Optional.7ef = temporary_storage
// CHECK:STDOUT:   %Optional.None.call: init %Optional.7ef = call %Optional.None.specific_fn() to %.loc14_41
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
// CHECK:STDOUT:   %ptr.5da: type = ptr_type cpp_void_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.94c: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.6de: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.cc0, @ptr.as.OptionalStorage.impl(cpp_void_type) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.5da, (%OptionalStorage.impl_witness.6de) [concrete]
// CHECK:STDOUT:   %Optional.2c0: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.aab: type = pattern_type %Optional.2c0 [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Optional.2c0, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.634: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.34c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b86: type = ptr_type %Optional.2c0 [concrete]
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
// CHECK:STDOUT:     .void = cpp_void_type
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Optional: %Optional.type = import_ref Core//prelude/types/optional, Optional, loaded [concrete = constants.%Optional.generic]
// CHECK:STDOUT:   %Core.import_ref.8c0: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.94c)]
// CHECK:STDOUT:   %Core.import_ref.566 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.cd4 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6db = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5a7 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.cc0 = impl_witness_table (%Core.import_ref.8c0, %Core.import_ref.566, %Core.import_ref.cd4, %Core.import_ref.6db, %Core.import_ref.5a7), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.5da, (constants.%OptionalStorage.impl_witness.6de) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10: %OptionalStorage.type = converted constants.%ptr.5da, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.2c0]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type.aab = value_binding_pattern output [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc10_50.1: ref %Optional.2c0 = temporary_storage
// CHECK:STDOUT:   %foo.call: init %Optional.2c0 = call imports.%foo.decl() to %.loc10_50.1
// CHECK:STDOUT:   %.loc10_38.1: type = splice_block %Optional [concrete = constants.%Optional.2c0] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc10_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, cpp_void_type [concrete = cpp_void_type]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.5da]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.6de) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10_38.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.2c0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_50.2: ref %Optional.2c0 = temporary %.loc10_50.1, %foo.call
// CHECK:STDOUT:   %.loc10_50.3: %Optional.2c0 = bind_value %.loc10_50.2
// CHECK:STDOUT:   %output: %Optional.2c0 = value_binding output, %.loc10_50.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_50.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.634
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_50.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.b86 = addr_of %.loc10_50.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.5da: type = ptr_type cpp_void_type [concrete]
// CHECK:STDOUT:   %ptr.99a: type = ptr_type %ptr.5da [concrete]
// CHECK:STDOUT:   %pattern_type.218: type = pattern_type %ptr.99a [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.type: type = cpp_overload_set_type @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.value: %Return.cpp_overload_set.type = cpp_overload_set_value @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.type: type = fn_type @Return [concrete]
// CHECK:STDOUT:   %Return: %Return.type = struct_value () [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.type: type = cpp_overload_set_type @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.value: %Invoke.cpp_overload_set.type = cpp_overload_set_value @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.type: type = fn_type @Invoke [concrete]
// CHECK:STDOUT:   %Invoke: %Invoke.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.99a, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6e8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fbf: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6e8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.7e4: type = ptr_type %ptr.99a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = cpp_void_type
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
// CHECK:STDOUT:     %non_nullable_pointer.patt: %pattern_type.218 = ref_binding_pattern non_nullable_pointer [concrete]
// CHECK:STDOUT:     %non_nullable_pointer.var_patt: %pattern_type.218 = var_pattern %non_nullable_pointer.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer.var: ref %ptr.99a = var %non_nullable_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return.ref: %Return.cpp_overload_set.type = name_ref Return, imports.%Return.cpp_overload_set.value [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.call: init %ptr.99a = call imports.%Return.decl()
// CHECK:STDOUT:   assign %non_nullable_pointer.var, %Return.call
// CHECK:STDOUT:   %.loc11: type = splice_block %ptr.loc11_38 [concrete = constants.%ptr.99a] {
// CHECK:STDOUT:     %Cpp.ref.loc11_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, cpp_void_type [concrete = cpp_void_type]
// CHECK:STDOUT:     %ptr.loc11_37: type = ptr_type %void.ref [concrete = constants.%ptr.5da]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %ptr.loc11_37 [concrete = constants.%ptr.99a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_nullable_pointer: ref %ptr.99a = ref_binding non_nullable_pointer, %non_nullable_pointer.var
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke.ref: %Invoke.cpp_overload_set.type = name_ref Invoke, imports.%Invoke.cpp_overload_set.value [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %non_nullable_pointer.ref: ref %ptr.99a = name_ref non_nullable_pointer, %non_nullable_pointer
// CHECK:STDOUT:   %.loc12: %ptr.99a = bind_value %non_nullable_pointer.ref
// CHECK:STDOUT:   %Invoke.call: init %empty_tuple.type = call imports.%Invoke.decl(%.loc12)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %non_nullable_pointer.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fbf
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %non_nullable_pointer.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.7e4 = addr_of %non_nullable_pointer.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_nullable_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %const: type = const_type cpp_void_type [concrete]
// CHECK:STDOUT:   %ptr.58e: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.dc1: type = pattern_type %ptr.58e [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.type: type = cpp_overload_set_type @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.cpp_overload_set.value: %Return.cpp_overload_set.type = cpp_overload_set_value @Return.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Return.type: type = fn_type @Return [concrete]
// CHECK:STDOUT:   %Return: %Return.type = struct_value () [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.type: type = cpp_overload_set_type @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.cpp_overload_set.value: %Invoke.cpp_overload_set.type = cpp_overload_set_value @Invoke.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Invoke.type: type = fn_type @Invoke [concrete]
// CHECK:STDOUT:   %Invoke: %Invoke.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.58e, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.97b: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.ed6: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.97b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.66d: type = ptr_type %ptr.58e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = cpp_void_type
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
// CHECK:STDOUT:     %const_void_pointer.patt: %pattern_type.dc1 = ref_binding_pattern const_void_pointer [concrete]
// CHECK:STDOUT:     %const_void_pointer.var_patt: %pattern_type.dc1 = var_pattern %const_void_pointer.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %const_void_pointer.var: ref %ptr.58e = var %const_void_pointer.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Return.ref: %Return.cpp_overload_set.type = name_ref Return, imports.%Return.cpp_overload_set.value [concrete = constants.%Return.cpp_overload_set.value]
// CHECK:STDOUT:   %Return.call: init %ptr.58e = call imports.%Return.decl()
// CHECK:STDOUT:   assign %const_void_pointer.var, %Return.call
// CHECK:STDOUT:   %.loc11: type = splice_block %ptr [concrete = constants.%ptr.58e] {
// CHECK:STDOUT:     %Cpp.ref.loc11_33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, cpp_void_type [concrete = cpp_void_type]
// CHECK:STDOUT:     %const: type = const_type %void.ref [concrete = constants.%const]
// CHECK:STDOUT:     %ptr: type = ptr_type %const [concrete = constants.%ptr.58e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %const_void_pointer: ref %ptr.58e = ref_binding const_void_pointer, %const_void_pointer.var
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Invoke.ref: %Invoke.cpp_overload_set.type = name_ref Invoke, imports.%Invoke.cpp_overload_set.value [concrete = constants.%Invoke.cpp_overload_set.value]
// CHECK:STDOUT:   %const_void_pointer.ref: ref %ptr.58e = name_ref const_void_pointer, %const_void_pointer
// CHECK:STDOUT:   %.loc12: %ptr.58e = bind_value %const_void_pointer.ref
// CHECK:STDOUT:   %Invoke.call: init %empty_tuple.type = call imports.%Invoke.decl(%.loc12)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %const_void_pointer.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ed6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %const_void_pointer.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.66d = addr_of %const_void_pointer.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
