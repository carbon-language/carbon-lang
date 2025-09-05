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
};

// --- use_struct_fields.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

fn F(s: Cpp.Struct) -> (i32, i32, i32) {
  //@dump-sem-ir-begin
  return (s.a, s.b, *s.p);
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
  // Nullable pointers are not supported.
  int *is_nullable;
  // References are not supported.
  int &is_reference;
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
  // CHECK:STDERR: fail_use_unsupported_members.carbon:[[@LINE+8]]:16: note: in `Cpp` name lookup for `is_volatile` [InCppNameLookup]
  // CHECK:STDERR:   let a: i32 = m->is_volatile;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_use_unsupported_members.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./unsupported_members.h:6:8: error: semantics TODO: `Unsupported: field declaration has unhandled type or kind` [SemanticsTodo]
  // CHECK:STDERR:   int *is_nullable;
  // CHECK:STDERR:        ^
  let a: i32 = m->is_volatile;

  // CHECK:STDERR: fail_use_unsupported_members.carbon:[[@LINE+8]]:17: note: in `Cpp` name lookup for `is_nullable` [InCppNameLookup]
  // CHECK:STDERR:   let b: i32 = *m->is_nullable;
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_use_unsupported_members.carbon:[[@LINE-17]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./unsupported_members.h:8:8: error: semantics TODO: `Unsupported: field declaration has unhandled type or kind` [SemanticsTodo]
  // CHECK:STDERR:   int &is_reference;
  // CHECK:STDERR:        ^
  let b: i32 = *m->is_nullable;

  // CHECK:STDERR: fail_use_unsupported_members.carbon:[[@LINE+4]]:16: note: in `Cpp` name lookup for `is_reference` [InCppNameLookup]
  // CHECK:STDERR:   let c: i32 = m->is_reference;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let c: i32 = m->is_reference;
}

// CHECK:STDOUT: --- use_struct_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Struct: type = class_type @Struct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %Struct.elem.86b: type = unbound_element_type %Struct, %i32 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Struct.elem.765: type = unbound_element_type %Struct, %ptr.235 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %Struct) -> %return.param: %tuple.type.189 {
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
// CHECK:STDOUT:   %.loc8_25.1: %tuple.type.189 = tuple_literal (%.loc8_12.2, %.loc8_17.2, %.loc8_21.1)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc8_25.2: init %i32 = initialize_from %.loc8_12.2 to %tuple.elem0
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc8_25.3: init %i32 = initialize_from %.loc8_17.2 to %tuple.elem1
// CHECK:STDOUT:   %.loc8_21.2: %i32 = bind_value %.loc8_21.1
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %return, element2
// CHECK:STDOUT:   %.loc8_25.4: init %i32 = initialize_from %.loc8_21.2 to %tuple.elem2
// CHECK:STDOUT:   %.loc8_25.5: init %tuple.type.189 = tuple_init (%.loc8_25.2, %.loc8_25.3, %.loc8_25.4) to %return
// CHECK:STDOUT:   %.loc8_26: init %tuple.type.189 = converted %.loc8_25.1, %.loc8_25.5
// CHECK:STDOUT:   return %.loc8_26 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_union_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Union: type = class_type @Union [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Union.elem.041: type = unbound_element_type %Union, %i32 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Union.elem.92a: type = unbound_element_type %Union, %ptr.235 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %Union = name_ref u, %u
// CHECK:STDOUT:   %a.ref: %Union.elem.041 = name_ref a, @Union.%.1 [concrete = @Union.%.1]
// CHECK:STDOUT:   %.loc8_11.1: ref %i32 = class_element_access %u.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: %i32 = bind_value %.loc8_11.1
// CHECK:STDOUT:   return %.loc8_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%u.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %Union = name_ref u, %u
// CHECK:STDOUT:   %b.ref: %Union.elem.041 = name_ref b, @Union.%.2 [concrete = @Union.%.2]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %u.ref, element1
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   return %.loc14_11.2
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
// CHECK:STDOUT:   return %.loc20_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non_bitfields_in_type_with_bitfields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Struct: type = class_type @Struct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Struct.elem: type = unbound_element_type %Struct, %i32 [concrete]
// CHECK:STDOUT:   %Union: type = class_type @Union [concrete]
// CHECK:STDOUT:   %Union.elem: type = unbound_element_type %Union, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %Struct) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Struct = name_ref s, %s
// CHECK:STDOUT:   %a.ref: %Struct.elem = name_ref a, @Struct.%.1 [concrete = @Struct.%.1]
// CHECK:STDOUT:   %.loc8_11.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: %i32 = bind_value %.loc8_11.1
// CHECK:STDOUT:   return %.loc8_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Union = name_ref s, %s
// CHECK:STDOUT:   %c.ref: %Union.elem = name_ref c, @Union.%.1 [concrete = @Union.%.1]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   return %.loc14_11.2
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
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Union = name_ref s, %s
// CHECK:STDOUT:   %d.ref: <error> = name_ref d, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
