// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/field.carbon

// --- struct.h

struct Struct {
  int a;
  int b;
  int* _Nonnull p;
  int* _Nullable q;
  int& r;
};

// --- use_struct_fields.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

fn F(s: Cpp.Struct) -> (i32, i32, i32, i32, i32) {
  //@dump-sem-ir-begin
  return (s.a, s.b, *s.p, *s.q.Get(), *s.r);
  //@dump-sem-ir-end
}

// --- union.h

union Union {
  int a;
  int b;
  int* _Nonnull p;
};

// --- use_union_fields.carbon

library "[[@TEST_NAME]]";

import Cpp library "union.h";

fn F(u: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  return u.a;
  //@dump-sem-ir-end
}

fn G(u: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  return u.b;
  //@dump-sem-ir-end
}

fn H(u: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  return *u.p;
  //@dump-sem-ir-end
}

// --- anon_struct_union.h

struct A {
  union {
    struct {
      short a_0;
      short b_2;
      int c_4;
    };
    struct {
      char d_0;
      char e_1;
      int f_4;
      short g_8;
    };
  };
  int h_12;
};

// --- use_anon_struct_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "anon_struct_union.h";

fn GetF(a: Cpp.A) -> i32 { return a.f_4; }

// --- with_bitfields.h

struct Struct {
  int a;
  int b : 2;
};

union Union {
  int c;
  int d : 3;
};

// --- use_non_bitfields_in_type_with_bitfields.carbon

library "[[@TEST_NAME]]";

import Cpp library "with_bitfields.h";

fn F(s: Cpp.Struct) -> i32 {
  //@dump-sem-ir-begin
  return s.a;
  //@dump-sem-ir-end
}

fn G(s: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  return s.c;
  //@dump-sem-ir-end
}

// --- fail_todo_use_bitfields.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./with_bitfields.h:4:7: error: semantics TODO: `Unsupported: field declaration has unhandled type or kind` [SemanticsTodo]
// CHECK:STDERR:   int b : 2;
// CHECK:STDERR:       ^
import Cpp library "with_bitfields.h";

fn F(s: Cpp.Struct) -> i32 {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+8]]:10: note: in `Cpp` name lookup for `b` [InCppNameLookup]
  // CHECK:STDERR:   return s.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./with_bitfields.h:9:7: error: semantics TODO: `Unsupported: field declaration has unhandled type or kind` [SemanticsTodo]
  // CHECK:STDERR:   int d : 3;
  // CHECK:STDERR:       ^
  return s.b;
  //@dump-sem-ir-end
}

fn G(s: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+4]]:10: note: in `Cpp` name lookup for `d` [InCppNameLookup]
  // CHECK:STDERR:   return s.d;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return s.d;
  //@dump-sem-ir-end
}

// --- unsupported_members.h

struct UnsupportedMembers {
  // Volatile is not supported.
  volatile int is_volatile;
  // But this should be fine.
  int integer;
};

void Consume(UnsupportedMembers);

// --- supported_members_in_type_with_unsupported_members.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_members.h";

fn Test(m: Cpp.UnsupportedMembers*) {
  // OK: Can still access supported members.
  m->integer = 42;
  // OK: Can still pass the class to a C++ function.
  Cpp.Consume(*m);
}

// --- fail_use_unsupported_members.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_use_unsupported_members.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./unsupported_members.h:4:16: error: semantics TODO: `Unsupported: field declaration has unhandled type or kind` [SemanticsTodo]
// CHECK:STDERR:   volatile int is_volatile;
// CHECK:STDERR:                ^
import Cpp library "unsupported_members.h";

fn Test(m: Cpp.UnsupportedMembers*) {
  // CHECK:STDERR: fail_use_unsupported_members.carbon:[[@LINE+4]]:16: note: in `Cpp` name lookup for `is_volatile` [InCppNameLookup]
  // CHECK:STDERR:   let a: i32 = m->is_volatile;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: i32 = m->is_volatile;
}

// CHECK:STDOUT: --- use_struct_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Struct: type = class_type @Struct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.a78: type = tuple_type (%i32, %i32, %i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %Struct.elem.86b: type = unbound_element_type %Struct, %i32 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Struct.elem.765: type = unbound_element_type %Struct, %ptr.235 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.76d: %OptionalStorage.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.11d: type = fn_type @Optional.Get, @Optional(%T.76d) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bb7: %Optional.Get.type.11d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.94c: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Get.type.d23: type = fn_type @ptr.as.OptionalStorage.impl.Get, @ptr.as.OptionalStorage.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Get.711: %ptr.as.OptionalStorage.impl.Get.type.d23 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.7fc: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.efb, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet: %OptionalStorage.type = facet_value %ptr.235, (%OptionalStorage.impl_witness.7fc) [concrete]
// CHECK:STDOUT:   %Optional.884: type = class_type @Optional, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Struct.elem.98c: type = unbound_element_type %Struct, %Optional.884 [concrete]
// CHECK:STDOUT:   %const.12f: type = const_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Struct.elem.93e: type = unbound_element_type %Struct, %const.12f [concrete]
// CHECK:STDOUT:   %Optional.Get.type.524: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Optional.Get.1ff: %Optional.Get.type.524 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.1ff, @Optional.Get(%OptionalStorage.facet) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c49: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c49 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a3d: @Optional.%Optional.Get.type (%Optional.Get.type.11d) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bb7)]
// CHECK:STDOUT:   %Core.import_ref.8c0: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.94c)]
// CHECK:STDOUT:   %Core.import_ref.566 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.cd4 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6db = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b65: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Get.type (%ptr.as.OptionalStorage.impl.Get.type.d23) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Get (constants.%ptr.as.OptionalStorage.impl.Get.711)]
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.efb = impl_witness_table (%Core.import_ref.8c0, %Core.import_ref.566, %Core.import_ref.cd4, %Core.import_ref.6db, %Core.import_ref.b65), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %Struct) -> %return.param: %tuple.type.a78 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref.loc8_11: %Struct = name_ref s, %s
// CHECK:STDOUT:   %a.ref: %Struct.elem.86b = name_ref a, @Struct.%.1 [concrete = @Struct.%.1]
// CHECK:STDOUT:   %.loc8_12.1: ref %i32 = class_element_access %s.ref.loc8_11, element0
// CHECK:STDOUT:   %.loc8_12.2: %i32 = bind_value %.loc8_12.1
// CHECK:STDOUT:   %s.ref.loc8_16: %Struct = name_ref s, %s
// CHECK:STDOUT:   %b.ref: %Struct.elem.86b = name_ref b, @Struct.%.2 [concrete = @Struct.%.2]
// CHECK:STDOUT:   %.loc8_17.1: ref %i32 = class_element_access %s.ref.loc8_16, element1
// CHECK:STDOUT:   %.loc8_17.2: %i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %s.ref.loc8_22: %Struct = name_ref s, %s
// CHECK:STDOUT:   %p.ref: %Struct.elem.765 = name_ref p, @Struct.%.3 [concrete = @Struct.%.3]
// CHECK:STDOUT:   %.loc8_23.1: ref %ptr.235 = class_element_access %s.ref.loc8_22, element2
// CHECK:STDOUT:   %.loc8_23.2: %ptr.235 = bind_value %.loc8_23.1
// CHECK:STDOUT:   %.loc8_21.1: ref %i32 = deref %.loc8_23.2
// CHECK:STDOUT:   %s.ref.loc8_28: %Struct = name_ref s, %s
// CHECK:STDOUT:   %q.ref: %Struct.elem.98c = name_ref q, @Struct.%.5 [concrete = @Struct.%.5]
// CHECK:STDOUT:   %.loc8_29.1: ref %Optional.884 = class_element_access %s.ref.loc8_28, element3
// CHECK:STDOUT:   %.loc8_29.2: %Optional.884 = bind_value %.loc8_29.1
// CHECK:STDOUT:   %.loc8_31: %Optional.Get.type.524 = specific_constant imports.%Core.import_ref.a3d, @Optional(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Get.1ff]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.524 = name_ref Get, %.loc8_31 [concrete = constants.%Optional.Get.1ff]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc8_29.2, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_36: <bound method> = bound_method %.loc8_29.2, %Optional.Get.specific_fn
// CHECK:STDOUT:   %Optional.Get.call: init %ptr.235 = call %bound_method.loc8_36(%.loc8_29.2)
// CHECK:STDOUT:   %.loc8_36.1: %ptr.235 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc8_36.2: %ptr.235 = converted %Optional.Get.call, %.loc8_36.1
// CHECK:STDOUT:   %.loc8_27.1: ref %i32 = deref %.loc8_36.2
// CHECK:STDOUT:   %s.ref.loc8_40: %Struct = name_ref s, %s
// CHECK:STDOUT:   %r.ref: %Struct.elem.93e = name_ref r, @Struct.%.6 [concrete = @Struct.%.6]
// CHECK:STDOUT:   %.loc8_41.1: ref %const.12f = class_element_access %s.ref.loc8_40, element4
// CHECK:STDOUT:   %.loc8_41.2: %const.12f = bind_value %.loc8_41.1
// CHECK:STDOUT:   %.loc8_39.1: ref %i32 = deref %.loc8_41.2
// CHECK:STDOUT:   %.loc8_43.1: %tuple.type.a78 = tuple_literal (%.loc8_12.2, %.loc8_17.2, %.loc8_21.1, %.loc8_27.1, %.loc8_39.1)
// CHECK:STDOUT:   %impl.elem0.loc8_12: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc8_12.1: <bound method> = bound_method %.loc8_12.2, %impl.elem0.loc8_12
// CHECK:STDOUT:   %specific_fn.loc8_12: <specific function> = specific_function %impl.elem0.loc8_12, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_12.2: <bound method> = bound_method %.loc8_12.2, %specific_fn.loc8_12
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc8_12: init %i32 = call %bound_method.loc8_12.2(%.loc8_12.2)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc8_43.2: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc8_12 to %tuple.elem0
// CHECK:STDOUT:   %impl.elem0.loc8_17: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc8_17.1: <bound method> = bound_method %.loc8_17.2, %impl.elem0.loc8_17
// CHECK:STDOUT:   %specific_fn.loc8_17: <specific function> = specific_function %impl.elem0.loc8_17, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_17.2: <bound method> = bound_method %.loc8_17.2, %specific_fn.loc8_17
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc8_17: init %i32 = call %bound_method.loc8_17.2(%.loc8_17.2)
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc8_43.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc8_17 to %tuple.elem1
// CHECK:STDOUT:   %.loc8_21.2: %i32 = bind_value %.loc8_21.1
// CHECK:STDOUT:   %impl.elem0.loc8_21: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc8_21.1: <bound method> = bound_method %.loc8_21.2, %impl.elem0.loc8_21
// CHECK:STDOUT:   %specific_fn.loc8_21: <specific function> = specific_function %impl.elem0.loc8_21, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_21.2: <bound method> = bound_method %.loc8_21.2, %specific_fn.loc8_21
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc8_21: init %i32 = call %bound_method.loc8_21.2(%.loc8_21.2)
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %return, element2
// CHECK:STDOUT:   %.loc8_43.4: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc8_21 to %tuple.elem2
// CHECK:STDOUT:   %.loc8_27.2: %i32 = bind_value %.loc8_27.1
// CHECK:STDOUT:   %impl.elem0.loc8_27: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc8_27.1: <bound method> = bound_method %.loc8_27.2, %impl.elem0.loc8_27
// CHECK:STDOUT:   %specific_fn.loc8_27: <specific function> = specific_function %impl.elem0.loc8_27, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_27.2: <bound method> = bound_method %.loc8_27.2, %specific_fn.loc8_27
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc8_27: init %i32 = call %bound_method.loc8_27.2(%.loc8_27.2)
// CHECK:STDOUT:   %tuple.elem3: ref %i32 = tuple_access %return, element3
// CHECK:STDOUT:   %.loc8_43.5: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc8_27 to %tuple.elem3
// CHECK:STDOUT:   %.loc8_39.2: %i32 = bind_value %.loc8_39.1
// CHECK:STDOUT:   %impl.elem0.loc8_39: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc8_39.1: <bound method> = bound_method %.loc8_39.2, %impl.elem0.loc8_39
// CHECK:STDOUT:   %specific_fn.loc8_39: <specific function> = specific_function %impl.elem0.loc8_39, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_39.2: <bound method> = bound_method %.loc8_39.2, %specific_fn.loc8_39
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc8_39: init %i32 = call %bound_method.loc8_39.2(%.loc8_39.2)
// CHECK:STDOUT:   %tuple.elem4: ref %i32 = tuple_access %return, element4
// CHECK:STDOUT:   %.loc8_43.6: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc8_39 to %tuple.elem4
// CHECK:STDOUT:   %.loc8_43.7: init %tuple.type.a78 = tuple_init (%.loc8_43.2, %.loc8_43.3, %.loc8_43.4, %.loc8_43.5, %.loc8_43.6) to %return
// CHECK:STDOUT:   %.loc8_44: init %tuple.type.a78 = converted %.loc8_43.1, %.loc8_43.7
// CHECK:STDOUT:   return %.loc8_44 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_union_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Union: type = class_type @Union [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Union.elem.041: type = unbound_element_type %Union, %i32 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Union.elem.92a: type = unbound_element_type %Union, %ptr.235 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %Union = name_ref u, %u
// CHECK:STDOUT:   %a.ref: %Union.elem.041 = name_ref a, @Union.%.1 [concrete = @Union.%.1]
// CHECK:STDOUT:   %.loc8_11.1: ref %i32 = class_element_access %u.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: %i32 = bind_value %.loc8_11.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc8_11.2(%.loc8_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%u.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %Union = name_ref u, %u
// CHECK:STDOUT:   %b.ref: %Union.elem.041 = name_ref b, @Union.%.2 [concrete = @Union.%.2]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %u.ref, element1
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%u.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %Union = name_ref u, %u
// CHECK:STDOUT:   %p.ref: %Union.elem.92a = name_ref p, @Union.%.3 [concrete = @Union.%.3]
// CHECK:STDOUT:   %.loc20_12.1: ref %ptr.235 = class_element_access %u.ref, element2
// CHECK:STDOUT:   %.loc20_12.2: %ptr.235 = bind_value %.loc20_12.1
// CHECK:STDOUT:   %.loc20_10.1: ref %i32 = deref %.loc20_12.2
// CHECK:STDOUT:   %.loc20_10.2: %i32 = bind_value %.loc20_10.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc20_10.1: <bound method> = bound_method %.loc20_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_10.2: <bound method> = bound_method %.loc20_10.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc20_10.2(%.loc20_10.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non_bitfields_in_type_with_bitfields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Struct: type = class_type @Struct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Struct.elem: type = unbound_element_type %Struct, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Union: type = class_type @Union [concrete]
// CHECK:STDOUT:   %Union.elem: type = unbound_element_type %Union, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %Struct) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Struct = name_ref s, %s
// CHECK:STDOUT:   %a.ref: %Struct.elem = name_ref a, @Struct.%.1 [concrete = @Struct.%.1]
// CHECK:STDOUT:   %.loc8_11.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: %i32 = bind_value %.loc8_11.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc8_11.2(%.loc8_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Union = name_ref s, %s
// CHECK:STDOUT:   %c.ref: %Union.elem = name_ref c, @Union.%.1 [concrete = @Union.%.1]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_bitfields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Struct: type = class_type @Struct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Union: type = class_type @Union [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %Struct) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Struct = name_ref s, %s
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Union = name_ref s, %s
// CHECK:STDOUT:   %d.ref: <error> = name_ref d, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
