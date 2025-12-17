// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/constructor.carbon

// ============================================================================
// Default constructor
// ============================================================================

// --- default.h

class C {
 public:
  C();
};

// --- import_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c: Cpp.C = Cpp.C.C();
  //@dump-sem-ir-end
}

// ============================================================================
// Non default constructor
// ============================================================================

// --- non_default.h

class C {
 public:
  C(int x, int y);
};

// --- import_non_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_default.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c: Cpp.C = Cpp.C.C(123, 456);
  //@dump-sem-ir-end
}

// ============================================================================
// Multiple constructors
// ============================================================================

// --- multiple.h

class C {
 public:
  C();
  C(int x, int y);
};

// --- import_multiple.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.C = Cpp.C.C();
  let unused c2: Cpp.C = Cpp.C.C(123, 456);
  //@dump-sem-ir-end
}

// ============================================================================
// Constructor with default arguments
// ============================================================================

// --- default_arguments.h

class C {
 public:
  C(int x, int y = 8);
};

// --- import_default_arguments.carbon

library "[[@TEST_NAME]]";

import Cpp library "default_arguments.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.C = Cpp.C.C(8, 9);
  let unused c2: Cpp.C = Cpp.C.C(8);
  //@dump-sem-ir-end
}

// ============================================================================
// Constructor templates
// ============================================================================

// --- template.h

class C {
 public:
  C(int);
  template<typename T> C(T);
};

// --- import_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "template.h";

fn F() {
  //@dump-sem-ir-begin
  var c1: Cpp.C = Cpp.C.C(123);
  var unused c2: Cpp.C = Cpp.C.C(true);
  var unused c3: Cpp.C = Cpp.C.C(&c1);
  //@dump-sem-ir-end
}

// ============================================================================
// Deleted constructor
// ============================================================================

// --- deleted.h

class C {
 public:
  C() = delete;
};

class D {
 public:
  D(int) = delete;
  D(short);
};

// --- fail_import_deleted.carbon

library "[[@TEST_NAME]]";

import Cpp library "deleted.h";

fn F() {
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+8]]:33: error: call to deleted function 'C' [CppInteropParseError]
  // CHECK:STDERR:    15 |   let unused c: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:       |                                 ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:4:3: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     4 |   C() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  let unused c: Cpp.C = Cpp.C.C();

  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+12]]:34: error: call to deleted function 'D' [CppInteropParseError]
  // CHECK:STDERR:    29 |   let unused d: Cpp.D = Cpp.D.D(0);
  // CHECK:STDERR:       |                                  ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:9:3: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     9 |   D(int) = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-20]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:10:3: note: candidate constructor [CppInteropParseNote]
  // CHECK:STDERR:    10 |   D(short);
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  let unused d: Cpp.D = Cpp.D.D(0);
}

// ============================================================================
// Implicit single argument constructor
// ============================================================================

// --- implicit_single_argument.h

class C {
 public:
  C(int x);
};

// --- fail_todo_import_implicit_single_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_single_argument.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.C = Cpp.C.C(8);
  // CHECK:STDERR: fail_todo_import_implicit_single_argument.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `i32` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let unused c2: Cpp.C = 8 as i32;
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR: fail_todo_import_implicit_single_argument.carbon:[[@LINE+4]]:26: note: type `i32` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let unused c2: Cpp.C = 8 as i32;
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR:
  let unused c2: Cpp.C = 8 as i32;
  //@dump-sem-ir-end
}

// ============================================================================
// Implicit multi arguments constructor
// ============================================================================

// --- implicit_multi_arguments.h

class C {
 public:
  C(int x, int y = 8);
};

// --- fail_todo_import_implicit_multi_arguments.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_multi_arguments.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.C = Cpp.C.C(8, 9);
  let unused c2: Cpp.C = Cpp.C.C(8);
  // CHECK:STDERR: fail_todo_import_implicit_multi_arguments.carbon:[[@LINE+7]]:26: error: cannot implicitly convert expression of type `i32` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let unused c3: Cpp.C = 8 as i32;
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR: fail_todo_import_implicit_multi_arguments.carbon:[[@LINE+4]]:26: note: type `i32` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let unused c3: Cpp.C = 8 as i32;
  // CHECK:STDERR:                          ^~~~~~~~
  // CHECK:STDERR:
  let unused c3: Cpp.C = 8 as i32;
  //@dump-sem-ir-end
}

// ============================================================================
// Trying to call a constructor of an incomplete class
// ============================================================================

// --- incomplete.h

class C;

// --- fail_import_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
   // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE+8]]:4: error: member access into incomplete class `Cpp.C` [QualifiedExprInIncompleteClassScope]
   // CHECK:STDERR:    Cpp.C.C();
   // CHECK:STDERR:    ^~~~~~~
   // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
   // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
   // CHECK:STDERR: class C;
   // CHECK:STDERR:       ^
   // CHECK:STDERR:
   Cpp.C.C();
}

// CHECK:STDOUT: --- import_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.217 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_28: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_30: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_33.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_33.2: init %C = in_place_init %C__carbon_thunk.call, %.loc8_33.1
// CHECK:STDOUT:   %.loc8_20: type = splice_block %C.ref.loc8_20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_33.3: ref %C = temporary %.loc8_33.1, %.loc8_33.2
// CHECK:STDOUT:   %.loc8_33.4: %C = acquire_value %.loc8_33.3
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc8_33.4
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc8_33.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc8_33.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7c9: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b1d: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc5: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %bound_method.6da: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.d17: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.217 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_28: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_30: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc8_41.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_33: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc8_33.1: <bound method> = bound_method %int_123, %impl.elem0.loc8_33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7c9]
// CHECK:STDOUT:   %specific_fn.loc8_33: <specific function> = specific_function %impl.elem0.loc8_33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_33.2: <bound method> = bound_method %int_123, %specific_fn.loc8_33 [concrete = constants.%bound_method.b1d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_33: init %i32 = call %bound_method.loc8_33.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc8_33.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_33 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc8_33.2: %i32 = converted %int_123, %.loc8_33.1 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %impl.elem0.loc8_38: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc8_38.1: <bound method> = bound_method %int_456, %impl.elem0.loc8_38 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc5]
// CHECK:STDOUT:   %specific_fn.loc8_38: <specific function> = specific_function %impl.elem0.loc8_38, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_38.2: <bound method> = bound_method %int_456, %specific_fn.loc8_38 [concrete = constants.%bound_method.6da]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_38: init %i32 = call %bound_method.loc8_38.2(%int_456) [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc8_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_38 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc8_38.2: %i32 = converted %int_456, %.loc8_38.1 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_41.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc8_33.2, %.loc8_38.2, %addr)
// CHECK:STDOUT:   %.loc8_41.2: init %C = in_place_init %C__carbon_thunk.call, %.loc8_41.1
// CHECK:STDOUT:   %.loc8_20: type = splice_block %C.ref.loc8_20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_41.3: ref %C = temporary %.loc8_41.1, %.loc8_41.2
// CHECK:STDOUT:   %.loc8_41.4: %C = acquire_value %.loc8_41.3
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc8_41.4
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc8_41.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc8_41.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7c9: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b1d: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc5: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %bound_method.6da: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.d17: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_34.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_34.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_34.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_34.1
// CHECK:STDOUT:   %.loc8_21: type = splice_block %C.ref.loc8_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_34.3: ref %C = temporary %.loc8_34.1, %.loc8_34.2
// CHECK:STDOUT:   %.loc8_34.4: %C = acquire_value %.loc8_34.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_34.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc9_42.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_123, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7c9]
// CHECK:STDOUT:   %specific_fn.loc9_34: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_123, %specific_fn.loc9_34 [concrete = constants.%bound_method.b1d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_34: init %i32 = call %bound_method.loc9_34.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_34 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_123, %.loc9_34.1 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %impl.elem0.loc9_39: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc9_39.1: <bound method> = bound_method %int_456, %impl.elem0.loc9_39 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc5]
// CHECK:STDOUT:   %specific_fn.loc9_39: <specific function> = specific_function %impl.elem0.loc9_39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_39.2: <bound method> = bound_method %int_456, %specific_fn.loc9_39 [concrete = constants.%bound_method.6da]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_39: init %i32 = call %bound_method.loc9_39.2(%int_456) [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc9_39.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_39 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc9_39.2: %i32 = converted %int_456, %.loc9_39.1 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_42.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc9_34.2, %.loc9_39.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_42.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_42.1
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_42.3: ref %C = temporary %.loc9_42.1, %.loc9_42.2
// CHECK:STDOUT:   %.loc9_42.4: %C = acquire_value %.loc9_42.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_42.4
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_42.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_42.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_34.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_34.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_default_arguments.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7ac: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.42b: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1f1: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %bound_method.0b2: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc8_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7ac]
// CHECK:STDOUT:   %specific_fn.loc8_34: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8_34 [concrete = constants.%bound_method.42b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_34: init %i32 = call %bound_method.loc8_34.2(%int_8.loc8) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_34 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_8.loc8, %.loc8_34.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc8_37: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_9, %impl.elem0.loc8_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1f1]
// CHECK:STDOUT:   %specific_fn.loc8_37: <specific function> = specific_function %impl.elem0.loc8_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_9, %specific_fn.loc8_37 [concrete = constants.%bound_method.0b2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_37: init %i32 = call %bound_method.loc8_37.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc8_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_37 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc8_37.2: %i32 = converted %int_9, %.loc8_37.1 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_38.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%.loc8_34.2, %.loc8_37.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_38.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_38.1
// CHECK:STDOUT:   %.loc8_21: type = splice_block %C.ref.loc8_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_38.3: ref %C = temporary %.loc8_38.1, %.loc8_38.2
// CHECK:STDOUT:   %.loc8_38.4: %C = acquire_value %.loc8_38.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc9_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7ac]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.42b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_8.loc9) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_8.loc9, %.loc9_34.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc9_34.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_35.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_35.1
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_35.3: ref %C = temporary %.loc9_35.1, %.loc9_35.2
// CHECK:STDOUT:   %.loc9_35.4: %C = acquire_value %.loc9_35.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_35.4
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_35.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_35.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_38.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.665: <witness> = impl_witness imports.%Copy.impl_witness_table.920 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.665) [concrete]
// CHECK:STDOUT:   %.521: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.3: type = fn_type @C__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.3: %C__carbon_thunk.type.65f120.3 = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.3f5: %type_where = facet_value bool, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b70: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.3f5) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.653: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b70 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.588: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.920 = impl_witness_table (%Core.import_ref.588), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.3: %C__carbon_thunk.type.65f120.3 = fn_decl @C__carbon_thunk.3 [concrete = constants.%C__carbon_thunk.d98342.3] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = ref_binding_pattern c1 [concrete]
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %impl.elem0.loc8: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc8_27.1: <bound method> = bound_method %int_123, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_27.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_27.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc8_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc8_27.2: %i32 = converted %int_123, %.loc8_27.1 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%.loc8_27.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_30: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_3
// CHECK:STDOUT:   assign %c1.var, %.loc8_30
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc9_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %.loc9_34.1: ref bool = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.521 = impl_witness_access constants.%Copy.impl_witness.665, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %true, %impl.elem0.loc9 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc9_34.1(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_34.2: ref bool = temporary %.loc9_34.1, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc9_38.1: %ptr.bb2 = addr_of %.loc9_34.2
// CHECK:STDOUT:   %addr.loc9_38.2: %ptr.d9e = addr_of %.loc9_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%addr.loc9_38.1, %addr.loc9_38.2)
// CHECK:STDOUT:   %.loc9_38: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_3
// CHECK:STDOUT:   assign %c2.var, %.loc9_38
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.217 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc10_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %c1.ref: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %addr.loc10_34: %ptr.d9e = addr_of %c1.ref
// CHECK:STDOUT:   %.loc10_3: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %addr.loc10_37: %ptr.d9e = addr_of %.loc10_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.3(%addr.loc10_34, %addr.loc10_37)
// CHECK:STDOUT:   %.loc10_37: init %C = in_place_init %C__carbon_thunk.call.loc10, %.loc10_3
// CHECK:STDOUT:   assign %c3.var, %.loc10_37
// CHECK:STDOUT:   %.loc10_21: type = splice_block %C.ref.loc10_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = ref_binding c3, %c3.var
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %c3.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%c3.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_34.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.653
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %.loc9_34.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_34.2(%.loc9_34.2)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %c2.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%c2.var)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %c1.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%c1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_implicit_single_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.42b: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %As.type.dbd: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.be4: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.b5e: %Core.IntLiteral.as.As.impl.Convert.type.be4 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c45: <witness> = impl_witness imports.%As.impl_witness_table.026, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.b71: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.070: %Core.IntLiteral.as.As.impl.Convert.type.b71 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.dbd = facet_value Core.IntLiteral, (%As.impl_witness.c45) [concrete]
// CHECK:STDOUT:   %.507: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.070 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.070, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.819: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.452: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.be4) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.b5e)]
// CHECK:STDOUT:   %As.impl_witness_table.026 = impl_witness_table (%Core.import_ref.452), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc8_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.42b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_34.2(%int_8.loc8) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_8.loc8, %.loc8_34.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc8_34.2, %addr)
// CHECK:STDOUT:   %.loc8_35.2: init %C = in_place_init %C__carbon_thunk.call, %.loc8_35.1
// CHECK:STDOUT:   %.loc8_21: type = splice_block %C.ref.loc8_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_35.3: ref %C = temporary %.loc8_35.1, %.loc8_35.2
// CHECK:STDOUT:   %.loc8_35.4: %C = acquire_value %.loc8_35.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc16: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16: %.507 = impl_witness_access constants.%As.impl_witness.c45, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.070]
// CHECK:STDOUT:   %bound_method.loc16_28.1: <bound method> = bound_method %int_8.loc16, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_28.2: <bound method> = bound_method %int_8.loc16, %specific_fn.loc16 [concrete = constants.%bound_method.819]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc16_28.2(%int_8.loc16) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc16_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc16_28.2: %i32 = converted %int_8.loc16, %.loc16_28.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc16_21: type = splice_block %C.ref.loc16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_28.3: %C = converted %.loc16_28.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %c2: %C = value_binding c2, <error> [concrete = <error>]
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc8_35.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc8_35.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_implicit_multi_arguments.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7ac: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.42b: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1f1: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %bound_method.0b2: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.dbd: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.be4: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.b5e: %Core.IntLiteral.as.As.impl.Convert.type.be4 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c45: <witness> = impl_witness imports.%As.impl_witness_table.026, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.b71: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.070: %Core.IntLiteral.as.As.impl.Convert.type.b71 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.dbd = facet_value Core.IntLiteral, (%As.impl_witness.c45) [concrete]
// CHECK:STDOUT:   %.507: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.070 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.070, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.819: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.452: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.be4) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.b5e)]
// CHECK:STDOUT:   %As.impl_witness_table.026 = impl_witness_table (%Core.import_ref.452), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc8_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7ac]
// CHECK:STDOUT:   %specific_fn.loc8_34: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8_34 [concrete = constants.%bound_method.42b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_34: init %i32 = call %bound_method.loc8_34.2(%int_8.loc8) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_34 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_8.loc8, %.loc8_34.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc8_37: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_9, %impl.elem0.loc8_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1f1]
// CHECK:STDOUT:   %specific_fn.loc8_37: <specific function> = specific_function %impl.elem0.loc8_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_9, %specific_fn.loc8_37 [concrete = constants.%bound_method.0b2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_37: init %i32 = call %bound_method.loc8_37.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc8_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_37 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc8_37.2: %i32 = converted %int_9, %.loc8_37.1 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_38.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%.loc8_34.2, %.loc8_37.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_38.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_38.1
// CHECK:STDOUT:   %.loc8_21: type = splice_block %C.ref.loc8_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_38.3: ref %C = temporary %.loc8_38.1, %.loc8_38.2
// CHECK:STDOUT:   %.loc8_38.4: %C = acquire_value %.loc8_38.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc9_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7ac]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.42b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_8.loc9) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_8.loc9, %.loc9_34.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc9_34.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_35.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_35.1
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_35.3: ref %C = temporary %.loc9_35.1, %.loc9_35.2
// CHECK:STDOUT:   %.loc9_35.4: %C = acquire_value %.loc9_35.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc17: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17: %.507 = impl_witness_access constants.%As.impl_witness.c45, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.070]
// CHECK:STDOUT:   %bound_method.loc17_28.1: <bound method> = bound_method %int_8.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_28.2: <bound method> = bound_method %int_8.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.819]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc17_28.2(%int_8.loc17) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc17_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc17_28.2: %i32 = converted %int_8.loc17, %.loc17_28.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc17_21: type = splice_block %C.ref.loc17 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc17: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_28.3: %C = converted %.loc17_28.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %c3: %C = value_binding c3, <error> [concrete = <error>]
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_35.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_35.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_38.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
