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

// CHECK:STDOUT: --- non_nullable_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type custom_cpp_void_type [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = custom_cpp_void_type
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
// CHECK:STDOUT:   %ptr: type = ptr_type custom_cpp_void_type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = custom_cpp_void_type
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
// CHECK:STDOUT:     %void.ref: type = name_ref void, custom_cpp_void_type [concrete = custom_cpp_void_type]
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
// CHECK:STDOUT:   %ptr.948: type = ptr_type custom_cpp_void_type [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.76d: %OptionalStorage.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.94c: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.4fc: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.566: %ptr.as.OptionalStorage.impl.Some.type.4fc = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.890: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.377, @ptr.as.OptionalStorage.impl(custom_cpp_void_type) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.948, (%OptionalStorage.impl_witness.890) [concrete]
// CHECK:STDOUT:   %Optional.160: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.172: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.160)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.06d: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Optional.160) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert.1, @T.binding.as_type.as.ImplicitAs.impl.3a5(%T.76d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.6ac: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7a9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.7bf, @T.binding.as_type.as.ImplicitAs.impl.3a5(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.eac: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert.1, @T.binding.as_type.as.ImplicitAs.impl.3a5(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.901: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.eac = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.172 = facet_value %ptr.948, (%ImplicitAs.impl_witness.7a9) [concrete]
// CHECK:STDOUT:   %.e09: type = fn_type_with_self_type %ImplicitAs.Convert.type.06d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %T.binding.as_type.as.ImplicitAs.impl.Convert.901, @T.binding.as_type.as.ImplicitAs.impl.Convert.1(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Optional.160, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.009: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.783: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.009 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f60: type = ptr_type %Optional.160 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = custom_cpp_void_type
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
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_16.1: type = splice_block %Optional [concrete = constants.%Optional.160] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.948, (constants.%OptionalStorage.impl_witness.890) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %.loc10_16.2: %OptionalStorage.type = converted constants.%ptr.948, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.160]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e04: @T.binding.as_type.as.ImplicitAs.impl.3a5.%T.binding.as_type.as.ImplicitAs.impl.Convert.type (%T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.ImplicitAs.impl.3a5.%T.binding.as_type.as.ImplicitAs.impl.Convert (constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.6ac)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.7bf = impl_witness_table (%Core.import_ref.e04), @T.binding.as_type.as.ImplicitAs.impl.3a5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.948) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref: %ptr.948 = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.e09 = impl_witness_access constants.%ImplicitAs.impl_witness.7a9, element0 [concrete = constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.901]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %input.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.binding.as_type.as.ImplicitAs.impl.Convert.1(constants.%OptionalStorage.facet) [concrete = constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %input.ref, %specific_fn
// CHECK:STDOUT:   %.loc10_11.1: ref %Optional.160 = temporary_storage
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.160 = call %bound_method.loc10_11.2(%input.ref) to %.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.2: init %Optional.160 = converted %input.ref, %T.binding.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc10_11.3: ref %Optional.160 = temporary %.loc10_11.1, %.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.4: %Optional.160 = bind_value %.loc10_11.3
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%.loc10_11.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_11.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.783
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f60 = addr_of %.loc10_11.3
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
// CHECK:STDOUT:   %ptr.948: type = ptr_type custom_cpp_void_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.94c: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.e66: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.b1f: %ptr.as.OptionalStorage.impl.None.type.e66 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.2df: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.534, @ptr.as.OptionalStorage.impl(custom_cpp_void_type) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.948, (%OptionalStorage.impl_witness.2df) [concrete]
// CHECK:STDOUT:   %Optional.1e2: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.type.1b2: type = fn_type @Optional.None, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.None.02d: %Optional.None.type.1b2 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %Optional.None.02d, @Optional.None(%OptionalStorage.facet) [concrete]
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
// CHECK:STDOUT:     .void = custom_cpp_void_type
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
// CHECK:STDOUT:   %void.ref: type = name_ref void, custom_cpp_void_type [concrete = custom_cpp_void_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.948]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.2df) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %.loc14_34: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:   %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.1e2]
// CHECK:STDOUT:   %.loc14_35: %Optional.None.type.1b2 = specific_constant imports.%Core.import_ref.a6c, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.02d]
// CHECK:STDOUT:   %None.ref: %Optional.None.type.1b2 = name_ref None, %.loc14_35 [concrete = constants.%Optional.None.02d]
// CHECK:STDOUT:   %Optional.None.specific_fn: <specific function> = specific_function %None.ref, @Optional.None(constants.%OptionalStorage.facet) [concrete = constants.%Optional.None.specific_fn]
// CHECK:STDOUT:   %.loc14_41: ref %Optional.1e2 = temporary_storage
// CHECK:STDOUT:   %Optional.None.call: init %Optional.1e2 = call %Optional.None.specific_fn() to %.loc14_41
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
// CHECK:STDOUT:   %ptr.948: type = ptr_type custom_cpp_void_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.94c: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.767: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.cc0, @ptr.as.OptionalStorage.impl(custom_cpp_void_type) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.948, (%OptionalStorage.impl_witness.767) [concrete]
// CHECK:STDOUT:   %Optional.36c: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.efd: type = pattern_type %Optional.36c [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Optional.36c, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.9ba: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fe5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.9ba = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b1e: type = ptr_type %Optional.36c [concrete]
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
// CHECK:STDOUT:     .void = custom_cpp_void_type
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
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.948, (constants.%OptionalStorage.impl_witness.767) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10: %OptionalStorage.type = converted constants.%ptr.948, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.36c]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %output.patt: %pattern_type.efd = value_binding_pattern output [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10_42: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc10_50.1: ref %Optional.36c = temporary_storage
// CHECK:STDOUT:   %.loc10_50.2: ref %Optional.36c = temporary_storage
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %Optional.36c = call imports.%foo__carbon_thunk.decl() to %.loc10_50.2
// CHECK:STDOUT:   %.loc10_38.1: type = splice_block %Optional [concrete = constants.%Optional.36c] {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Optional.ref: %Optional.type = name_ref Optional, imports.%Core.Optional [concrete = constants.%Optional.generic]
// CHECK:STDOUT:     %Cpp.ref.loc10_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %void.ref: type = name_ref void, custom_cpp_void_type [concrete = custom_cpp_void_type]
// CHECK:STDOUT:     %ptr: type = ptr_type %void.ref [concrete = constants.%ptr.948]
// CHECK:STDOUT:     %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr, (constants.%OptionalStorage.impl_witness.767) [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %.loc10_38.2: %OptionalStorage.type = converted %ptr, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet]
// CHECK:STDOUT:     %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.36c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_50.3: ref %Optional.36c = temporary %.loc10_50.2, %foo__carbon_thunk.call
// CHECK:STDOUT:   %.loc10_50.4: %Optional.36c = bind_value %.loc10_50.3
// CHECK:STDOUT:   %output: %Optional.36c = value_binding output, %.loc10_50.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_50.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fe5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_50.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.b1e = addr_of %.loc10_50.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- wrapped.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.948: type = ptr_type custom_cpp_void_type [concrete]
// CHECK:STDOUT:   %ptr.1d5: type = ptr_type %ptr.948 [concrete]
// CHECK:STDOUT:   %NonNullableNonNullablePointer.cpp_overload_set.type: type = cpp_overload_set_type @NonNullableNonNullablePointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NonNullableNonNullablePointer.cpp_overload_set.value: %NonNullableNonNullablePointer.cpp_overload_set.type = cpp_overload_set_value @NonNullableNonNullablePointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NonNullableNonNullablePointer.type: type = fn_type @NonNullableNonNullablePointer [concrete]
// CHECK:STDOUT:   %NonNullableNonNullablePointer: %NonNullableNonNullablePointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonNullableNullablePointer.cpp_overload_set.type: type = cpp_overload_set_type @NonNullableNullablePointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NonNullableNullablePointer.cpp_overload_set.value: %NonNullableNullablePointer.cpp_overload_set.type = cpp_overload_set_value @NonNullableNullablePointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.76d: %OptionalStorage.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.94c: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.4fc: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.566: %ptr.as.OptionalStorage.impl.Some.type.4fc = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.70b: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.377, @ptr.as.OptionalStorage.impl(%ptr.948) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.243: %OptionalStorage.type = facet_value %ptr.1d5, (%OptionalStorage.impl_witness.70b) [concrete]
// CHECK:STDOUT:   %Optional.2ec: type = class_type @Optional, @Optional(%OptionalStorage.facet.243) [concrete]
// CHECK:STDOUT:   %NonNullableNullablePointer__carbon_thunk.type: type = fn_type @NonNullableNullablePointer__carbon_thunk [concrete]
// CHECK:STDOUT:   %NonNullableNullablePointer__carbon_thunk: %NonNullableNullablePointer__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e95: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.2ec)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.df2: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Optional.2ec) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert.1, @T.binding.as_type.as.ImplicitAs.impl.3a5(%T.76d) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.6ac: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.21e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.7bf, @T.binding.as_type.as.ImplicitAs.impl.3a5(%OptionalStorage.facet.243) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.type.39f: type = fn_type @T.binding.as_type.as.ImplicitAs.impl.Convert.1, @T.binding.as_type.as.ImplicitAs.impl.3a5(%OptionalStorage.facet.243) [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.d57: %T.binding.as_type.as.ImplicitAs.impl.Convert.type.39f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e95 = facet_value %ptr.1d5, (%ImplicitAs.impl_witness.21e) [concrete]
// CHECK:STDOUT:   %.12e: type = fn_type_with_self_type %ImplicitAs.Convert.type.df2, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %T.binding.as_type.as.ImplicitAs.impl.Convert.d57, @T.binding.as_type.as.ImplicitAs.impl.Convert.1(%OptionalStorage.facet.243) [concrete]
// CHECK:STDOUT:   %NullableNonNullablePointer.cpp_overload_set.type: type = cpp_overload_set_type @NullableNonNullablePointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NullableNonNullablePointer.cpp_overload_set.value: %NullableNonNullablePointer.cpp_overload_set.type = cpp_overload_set_value @NullableNonNullablePointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.890: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.377, @ptr.as.OptionalStorage.impl(custom_cpp_void_type) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.35b: %OptionalStorage.type = facet_value %ptr.948, (%OptionalStorage.impl_witness.890) [concrete]
// CHECK:STDOUT:   %Optional.160: type = class_type @Optional, @Optional(%OptionalStorage.facet.35b) [concrete]
// CHECK:STDOUT:   %ptr.f60: type = ptr_type %Optional.160 [concrete]
// CHECK:STDOUT:   %NullableNonNullablePointer.type: type = fn_type @NullableNonNullablePointer [concrete]
// CHECK:STDOUT:   %NullableNonNullablePointer: %NullableNonNullablePointer.type = struct_value () [concrete]
// CHECK:STDOUT:   %NullableNullablePointer.cpp_overload_set.type: type = cpp_overload_set_type @NullableNullablePointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NullableNullablePointer.cpp_overload_set.value: %NullableNullablePointer.cpp_overload_set.type = cpp_overload_set_value @NullableNullablePointer.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.722: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.377, @ptr.as.OptionalStorage.impl(%Optional.160) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.cf7: %OptionalStorage.type = facet_value %ptr.f60, (%OptionalStorage.impl_witness.722) [concrete]
// CHECK:STDOUT:   %Optional.d3c: type = class_type @Optional, @Optional(%OptionalStorage.facet.cf7) [concrete]
// CHECK:STDOUT:   %NullableNullablePointer__carbon_thunk.type: type = fn_type @NullableNullablePointer__carbon_thunk [concrete]
// CHECK:STDOUT:   %NullableNullablePointer__carbon_thunk: %NullableNullablePointer__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Optional.2ec, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.692: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.239: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.692 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.7cd: type = ptr_type %Optional.2ec [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .void = custom_cpp_void_type
// CHECK:STDOUT:     .NonNullableNonNullablePointer = %NonNullableNonNullablePointer.cpp_overload_set.value
// CHECK:STDOUT:     .NonNullableNullablePointer = %NonNullableNullablePointer.cpp_overload_set.value
// CHECK:STDOUT:     .NullableNonNullablePointer = %NullableNonNullablePointer.cpp_overload_set.value
// CHECK:STDOUT:     .NullableNullablePointer = %NullableNullablePointer.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonNullableNonNullablePointer.cpp_overload_set.value: %NonNullableNonNullablePointer.cpp_overload_set.type = cpp_overload_set_value @NonNullableNonNullablePointer.cpp_overload_set [concrete = constants.%NonNullableNonNullablePointer.cpp_overload_set.value]
// CHECK:STDOUT:   %NonNullableNonNullablePointer.decl: %NonNullableNonNullablePointer.type = fn_decl @NonNullableNonNullablePointer [concrete = constants.%NonNullableNonNullablePointer] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonNullableNullablePointer.cpp_overload_set.value: %NonNullableNullablePointer.cpp_overload_set.type = cpp_overload_set_value @NonNullableNullablePointer.cpp_overload_set [concrete = constants.%NonNullableNullablePointer.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.8c0: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.94c)]
// CHECK:STDOUT:   %Core.import_ref.566 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.637: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.4fc) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.566)]
// CHECK:STDOUT:   %Core.import_ref.6db = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5a7 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.377 = impl_witness_table (%Core.import_ref.8c0, %Core.import_ref.566, %Core.import_ref.637, %Core.import_ref.6db, %Core.import_ref.5a7), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %NonNullableNullablePointer__carbon_thunk.decl: %NonNullableNullablePointer__carbon_thunk.type = fn_decl @NonNullableNullablePointer__carbon_thunk [concrete = constants.%NonNullableNullablePointer__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc14_39.1: type = splice_block %Optional [concrete = constants.%Optional.2ec] {
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.1d5, (constants.%OptionalStorage.impl_witness.70b) [concrete = constants.%OptionalStorage.facet.243]
// CHECK:STDOUT:       %.loc14_39.2: %OptionalStorage.type = converted constants.%ptr.1d5, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.243]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.243) [concrete = constants.%Optional.2ec]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e04: @T.binding.as_type.as.ImplicitAs.impl.3a5.%T.binding.as_type.as.ImplicitAs.impl.Convert.type (%T.binding.as_type.as.ImplicitAs.impl.Convert.type.6f7) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.ImplicitAs.impl.3a5.%T.binding.as_type.as.ImplicitAs.impl.Convert (constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.6ac)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.7bf = impl_witness_table (%Core.import_ref.e04), @T.binding.as_type.as.ImplicitAs.impl.3a5 [concrete]
// CHECK:STDOUT:   %NullableNonNullablePointer.cpp_overload_set.value: %NullableNonNullablePointer.cpp_overload_set.type = cpp_overload_set_value @NullableNonNullablePointer.cpp_overload_set [concrete = constants.%NullableNonNullablePointer.cpp_overload_set.value]
// CHECK:STDOUT:   %NullableNonNullablePointer.decl: %NullableNonNullablePointer.type = fn_decl @NullableNonNullablePointer [concrete = constants.%NullableNonNullablePointer] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NullableNullablePointer.cpp_overload_set.value: %NullableNullablePointer.cpp_overload_set.type = cpp_overload_set_value @NullableNullablePointer.cpp_overload_set [concrete = constants.%NullableNullablePointer.cpp_overload_set.value]
// CHECK:STDOUT:   %NullableNullablePointer__carbon_thunk.decl: %NullableNullablePointer__carbon_thunk.type = fn_decl @NullableNullablePointer__carbon_thunk [concrete = constants.%NullableNullablePointer__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc16_36.1: type = splice_block %Optional.loc16_36.2 [concrete = constants.%Optional.d3c] {
// CHECK:STDOUT:       %OptionalStorage.facet.loc16_36.1: %OptionalStorage.type = facet_value constants.%ptr.948, (constants.%OptionalStorage.impl_witness.890) [concrete = constants.%OptionalStorage.facet.35b]
// CHECK:STDOUT:       %.loc16_36.2: %OptionalStorage.type = converted constants.%ptr.948, %OptionalStorage.facet.loc16_36.1 [concrete = constants.%OptionalStorage.facet.35b]
// CHECK:STDOUT:       %Optional.loc16_36.1: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.35b) [concrete = constants.%Optional.160]
// CHECK:STDOUT:       %OptionalStorage.facet.loc16_36.2: %OptionalStorage.type = facet_value constants.%ptr.f60, (constants.%OptionalStorage.impl_witness.722) [concrete = constants.%OptionalStorage.facet.cf7]
// CHECK:STDOUT:       %.loc16_36.3: %OptionalStorage.type = converted constants.%ptr.f60, %OptionalStorage.facet.loc16_36.2 [concrete = constants.%OptionalStorage.facet.cf7]
// CHECK:STDOUT:       %Optional.loc16_36.2: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.cf7) [concrete = constants.%Optional.d3c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%input.param: %ptr.1d5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NonNullableNonNullablePointer.ref: %NonNullableNonNullablePointer.cpp_overload_set.type = name_ref NonNullableNonNullablePointer, imports.%NonNullableNonNullablePointer.cpp_overload_set.value [concrete = constants.%NonNullableNonNullablePointer.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref.loc13: %ptr.1d5 = name_ref input, %input
// CHECK:STDOUT:   %NonNullableNonNullablePointer.call: init %empty_tuple.type = call imports.%NonNullableNonNullablePointer.decl(%input.ref.loc13)
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NonNullableNullablePointer.ref: %NonNullableNullablePointer.cpp_overload_set.type = name_ref NonNullableNullablePointer, imports.%NonNullableNullablePointer.cpp_overload_set.value [concrete = constants.%NonNullableNullablePointer.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref.loc14: %ptr.1d5 = name_ref input, %input
// CHECK:STDOUT:   %impl.elem0: %.12e = impl_witness_access constants.%ImplicitAs.impl_witness.21e, element0 [concrete = constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.d57]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %input.ref.loc14, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @T.binding.as_type.as.ImplicitAs.impl.Convert.1(constants.%OptionalStorage.facet.243) [concrete = constants.%T.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %input.ref.loc14, %specific_fn
// CHECK:STDOUT:   %.loc14_34.1: ref %Optional.2ec = temporary_storage
// CHECK:STDOUT:   %T.binding.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.2ec = call %bound_method.loc14_34.2(%input.ref.loc14) to %.loc14_34.1
// CHECK:STDOUT:   %.loc14_34.2: init %Optional.2ec = converted %input.ref.loc14, %T.binding.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc14_34.3: ref %Optional.2ec = temporary %.loc14_34.1, %.loc14_34.2
// CHECK:STDOUT:   %.loc14_34.4: %Optional.2ec = bind_value %.loc14_34.3
// CHECK:STDOUT:   %NonNullableNullablePointer__carbon_thunk.call: init %empty_tuple.type = call imports.%NonNullableNullablePointer__carbon_thunk.decl(%.loc14_34.4)
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NullableNonNullablePointer.ref: %NullableNonNullablePointer.cpp_overload_set.type = name_ref NullableNonNullablePointer, imports.%NullableNonNullablePointer.cpp_overload_set.value [concrete = constants.%NullableNonNullablePointer.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref.loc15: %ptr.1d5 = name_ref input, %input
// CHECK:STDOUT:   %.loc15_34.1: %ptr.f60 = as_compatible %input.ref.loc15
// CHECK:STDOUT:   %.loc15_34.2: %ptr.f60 = converted %input.ref.loc15, %.loc15_34.1
// CHECK:STDOUT:   %NullableNonNullablePointer.call: init %empty_tuple.type = call imports.%NullableNonNullablePointer.decl(%.loc15_34.2)
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NullableNullablePointer.ref: %NullableNullablePointer.cpp_overload_set.type = name_ref NullableNullablePointer, imports.%NullableNullablePointer.cpp_overload_set.value [concrete = constants.%NullableNullablePointer.cpp_overload_set.value]
// CHECK:STDOUT:   %input.ref.loc16: %ptr.1d5 = name_ref input, %input
// CHECK:STDOUT:   %.loc16: %Optional.d3c = converted %input.ref.loc16, <error> [concrete = <error>]
// CHECK:STDOUT:   %NullableNullablePointer__carbon_thunk.call: init %empty_tuple.type = call imports.%NullableNullablePointer__carbon_thunk.decl(<error>)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc14_34.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.239
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc14_34.3: <bound method> = bound_method %.loc14_34.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.7cd = addr_of %.loc14_34.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_34.3(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
