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
  let c: Cpp.C = Cpp.C.C();
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
  let c: Cpp.C = Cpp.C.C(123, 456);
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
  let c1: Cpp.C = Cpp.C.C();
  let c2: Cpp.C = Cpp.C.C(123, 456);
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
  let c1: Cpp.C = Cpp.C.C(8, 9);
  let c2: Cpp.C = Cpp.C.C(8);
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
  var c2: Cpp.C = Cpp.C.C(true);
  var c3: Cpp.C = Cpp.C.C(&c1);
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
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+8]]:26: error: call to deleted function 'C' [CppInteropParseError]
  // CHECK:STDERR:    15 |   let c: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:4:3: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     4 |   C() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  let c: Cpp.C = Cpp.C.C();

  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+12]]:27: error: call to deleted function 'D' [CppInteropParseError]
  // CHECK:STDERR:    29 |   let d: Cpp.D = Cpp.D.D(0);
  // CHECK:STDERR:       |                           ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:9:3: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     9 |   D(int) = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-20]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:10:3: note: candidate constructor [CppInteropParseNote]
  // CHECK:STDERR:    10 |   D(short);
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  let d: Cpp.D = Cpp.D.D(0);
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
  let c1: Cpp.C = Cpp.C.C(8);
  // CHECK:STDERR: fail_todo_import_implicit_single_argument.carbon:[[@LINE+7]]:19: error: cannot implicitly convert expression of type `i32` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let c2: Cpp.C = 8 as i32;
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR: fail_todo_import_implicit_single_argument.carbon:[[@LINE+4]]:19: note: type `i32` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c2: Cpp.C = 8 as i32;
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  let c2: Cpp.C = 8 as i32;
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
  let c1: Cpp.C = Cpp.C.C(8, 9);
  let c2: Cpp.C = Cpp.C.C(8);
  // CHECK:STDERR: fail_todo_import_implicit_multi_arguments.carbon:[[@LINE+7]]:19: error: cannot implicitly convert expression of type `i32` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let c3: Cpp.C = 8 as i32;
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR: fail_todo_import_implicit_multi_arguments.carbon:[[@LINE+4]]:19: note: type `i32` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let c3: Cpp.C = 8 as i32;
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  let c3: Cpp.C = 8 as i32;
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
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f77: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_23: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_26.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_26.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_26.2: init %C = in_place_init %C__carbon_thunk.call, %.loc8_26.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %C.ref.loc8_13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_26.3: ref %C = temporary %.loc8_26.1, %.loc8_26.2
// CHECK:STDOUT:   %.loc8_26.4: %C = acquire_value %.loc8_26.3
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc8_26.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_26.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_26.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%.loc8_26.3)
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e5: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.087: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c9b: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.0ea: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.d17: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f77: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.217 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_23: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc8_34.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_26: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_26.1: <bound method> = bound_method %int_123, %impl.elem0.loc8_26 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e5]
// CHECK:STDOUT:   %specific_fn.loc8_26: <specific function> = specific_function %impl.elem0.loc8_26, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_26.2: <bound method> = bound_method %int_123, %specific_fn.loc8_26 [concrete = constants.%bound_method.087]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_26: init %i32 = call %bound_method.loc8_26.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc8_26.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_26 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc8_26.2: %i32 = converted %int_123, %.loc8_26.1 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %impl.elem0.loc8_31: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %int_456, %impl.elem0.loc8_31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c9b]
// CHECK:STDOUT:   %specific_fn.loc8_31: <specific function> = specific_function %impl.elem0.loc8_31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %int_456, %specific_fn.loc8_31 [concrete = constants.%bound_method.0ea]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_31: init %i32 = call %bound_method.loc8_31.2(%int_456) [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc8_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_31 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc8_31.2: %i32 = converted %int_456, %.loc8_31.1 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_34.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc8_26.2, %.loc8_31.2, %addr)
// CHECK:STDOUT:   %.loc8_34.2: init %C = in_place_init %C__carbon_thunk.call, %.loc8_34.1
// CHECK:STDOUT:   %.loc8_13: type = splice_block %C.ref.loc8_13 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_13: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_34.3: ref %C = temporary %.loc8_34.1, %.loc8_34.2
// CHECK:STDOUT:   %.loc8_34.4: %C = acquire_value %.loc8_34.3
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc8_34.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_34.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_34: <bound method> = bound_method %.loc8_34.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_34(%.loc8_34.3)
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e5: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.087: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c9b: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.0ea: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.d17: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f77: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_27.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_27.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_27.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_27.3: ref %C = temporary %.loc8_27.1, %.loc8_27.2
// CHECK:STDOUT:   %.loc8_27.4: %C = acquire_value %.loc8_27.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc9_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9_27: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_27.1: <bound method> = bound_method %int_123, %impl.elem0.loc9_27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e5]
// CHECK:STDOUT:   %specific_fn.loc9_27: <specific function> = specific_function %impl.elem0.loc9_27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_27.2: <bound method> = bound_method %int_123, %specific_fn.loc9_27 [concrete = constants.%bound_method.087]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_27: init %i32 = call %bound_method.loc9_27.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc9_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_27 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc9_27.2: %i32 = converted %int_123, %.loc9_27.1 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %impl.elem0.loc9_32: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_32.1: <bound method> = bound_method %int_456, %impl.elem0.loc9_32 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c9b]
// CHECK:STDOUT:   %specific_fn.loc9_32: <specific function> = specific_function %impl.elem0.loc9_32, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_32.2: <bound method> = bound_method %int_456, %specific_fn.loc9_32 [concrete = constants.%bound_method.0ea]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_32: init %i32 = call %bound_method.loc9_32.2(%int_456) [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc9_32.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_32 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc9_32.2: %i32 = converted %int_456, %.loc9_32.1 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc9_27.2, %.loc9_32.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_35.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_35.1
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_35.3: ref %C = temporary %.loc9_35.1, %.loc9_35.2
// CHECK:STDOUT:   %.loc9_35.4: %C = acquire_value %.loc9_35.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_35.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_35.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_35.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9_35(%.loc9_35.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_27.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_27.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%.loc8_27.3)
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e6: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.0d0: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.268: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.e3f: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f77: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc8_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_27: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_27.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8_27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e6]
// CHECK:STDOUT:   %specific_fn.loc8_27: <specific function> = specific_function %impl.elem0.loc8_27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_27.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8_27 [concrete = constants.%bound_method.0d0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_27: init %i32 = call %bound_method.loc8_27.2(%int_8.loc8) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_27 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_27.2: %i32 = converted %int_8.loc8, %.loc8_27.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc8_30: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_30.1: <bound method> = bound_method %int_9, %impl.elem0.loc8_30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.268]
// CHECK:STDOUT:   %specific_fn.loc8_30: <specific function> = specific_function %impl.elem0.loc8_30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_30.2: <bound method> = bound_method %int_9, %specific_fn.loc8_30 [concrete = constants.%bound_method.e3f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_30: init %i32 = call %bound_method.loc8_30.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc8_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_30 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc8_30.2: %i32 = converted %int_9, %.loc8_30.1 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%.loc8_27.2, %.loc8_30.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_31.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_31.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: ref %C = temporary %.loc8_31.1, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %C = acquire_value %.loc8_31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc9_28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_27.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e6]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_27.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.0d0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_27.2(%int_8.loc9) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc9_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc9_27.2: %i32 = converted %int_8.loc9, %.loc9_27.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_28.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc9_27.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_28.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_28.1
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_28.3: ref %C = temporary %.loc9_28.1, %.loc9_28.2
// CHECK:STDOUT:   %.loc9_28.4: %C = acquire_value %.loc9_28.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_28.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_28.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9_28: <bound method> = bound_method %.loc9_28.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9_28(%.loc9_28.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_31: <bound method> = bound_method %.loc8_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_31(%.loc8_31.3)
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a56: <witness> = impl_witness imports.%Copy.impl_witness_table.189 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.a56) [concrete]
// CHECK:STDOUT:   %.05c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.3: type = fn_type @C__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.3: %C__carbon_thunk.type.65f120.3 = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.b21: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b21) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f77: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.3f5: %type_where = facet_value bool, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.694: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.3f5) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2db: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.694 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.cc5: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.189 = impl_witness_table (%Core.import_ref.cc5), @bool.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
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
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %.loc9_27.1: ref bool = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.05c = impl_witness_access constants.%Copy.impl_witness.a56, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9_27.1: <bound method> = bound_method %true, %impl.elem0.loc9 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc9_27.1(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_27.2: ref bool = temporary %.loc9_27.1, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc9_31.1: %ptr.bb2 = addr_of %.loc9_27.2
// CHECK:STDOUT:   %addr.loc9_31.2: %ptr.d9e = addr_of %.loc9_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%addr.loc9_31.1, %addr.loc9_31.2)
// CHECK:STDOUT:   %.loc9_31: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_3
// CHECK:STDOUT:   assign %c2.var, %.loc9_31
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.217 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc10_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %c1.ref: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %addr.loc10_27: %ptr.d9e = addr_of %c1.ref
// CHECK:STDOUT:   %.loc10_3: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %addr.loc10_30: %ptr.d9e = addr_of %.loc10_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.3(%addr.loc10_27, %addr.loc10_30)
// CHECK:STDOUT:   %.loc10_30: init %C = in_place_init %C__carbon_thunk.call.loc10, %.loc10_3
// CHECK:STDOUT:   assign %c3.var, %.loc10_30
// CHECK:STDOUT:   %.loc10_14: type = splice_block %C.ref.loc10_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = ref_binding c3, %c3.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %c3.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %c3.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%c3.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_27: <bound method> = bound_method %.loc9_27.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2db
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9_27.2: <bound method> = bound_method %.loc9_27.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_27: init %empty_tuple.type = call %bound_method.loc9_27.2(%.loc9_27.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_3: <bound method> = bound_method %c2.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9_3: <bound method> = bound_method %c2.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_3: init %empty_tuple.type = call %bound_method.loc9_3(%c2.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %c1.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_3: <bound method> = bound_method %c1.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_3(%c1.var)
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.0d0: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %As.type.dbd: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0fd: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.2b1: %Core.IntLiteral.as.As.impl.Convert.type.0fd = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bf0: <witness> = impl_witness imports.%As.impl_witness_table.251, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.11b: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.035: %Core.IntLiteral.as.As.impl.Convert.type.11b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.dbd = facet_value Core.IntLiteral, (%As.impl_witness.bf0) [concrete]
// CHECK:STDOUT:   %.edb: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.035 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.035, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.493: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f77: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.05d: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0fd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.2b1)]
// CHECK:STDOUT:   %As.impl_witness_table.251 = impl_witness_table (%Core.import_ref.05d), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc8_28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_27.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_27.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.0d0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_27.2(%int_8.loc8) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_27.2: %i32 = converted %int_8.loc8, %.loc8_27.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc8_28.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc8_27.2, %addr)
// CHECK:STDOUT:   %.loc8_28.2: init %C = in_place_init %C__carbon_thunk.call, %.loc8_28.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_28.3: ref %C = temporary %.loc8_28.1, %.loc8_28.2
// CHECK:STDOUT:   %.loc8_28.4: %C = acquire_value %.loc8_28.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_28.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc16: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16: %.edb = impl_witness_access constants.%As.impl_witness.bf0, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.035]
// CHECK:STDOUT:   %bound_method.loc16_21.1: <bound method> = bound_method %int_8.loc16, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_21.2: <bound method> = bound_method %int_8.loc16, %specific_fn.loc16 [concrete = constants.%bound_method.493]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc16_21.2(%int_8.loc16) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc16_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc16_21.2: %i32 = converted %int_8.loc16, %.loc16_21.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc16_14: type = splice_block %C.ref.loc16 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_21.3: %C = converted %.loc16_21.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %c2: %C = value_binding c2, <error> [concrete = <error>]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_28.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_28: <bound method> = bound_method %.loc8_28.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_28(%.loc8_28.3)
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e6: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.0d0: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.268: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.e3f: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.dbd: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0fd: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.2b1: %Core.IntLiteral.as.As.impl.Convert.type.0fd = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bf0: <witness> = impl_witness imports.%As.impl_witness_table.251, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.11b: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.035: %Core.IntLiteral.as.As.impl.Convert.type.11b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.dbd = facet_value Core.IntLiteral, (%As.impl_witness.bf0) [concrete]
// CHECK:STDOUT:   %.edb: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.035 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.035, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.493: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f77: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.248 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.05d: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0fd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.2b1)]
// CHECK:STDOUT:   %As.impl_witness_table.251 = impl_witness_table (%Core.import_ref.05d), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc8_31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_27: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_27.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8_27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e6]
// CHECK:STDOUT:   %specific_fn.loc8_27: <specific function> = specific_function %impl.elem0.loc8_27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_27.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8_27 [concrete = constants.%bound_method.0d0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_27: init %i32 = call %bound_method.loc8_27.2(%int_8.loc8) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_27 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc8_27.2: %i32 = converted %int_8.loc8, %.loc8_27.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc8_30: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_30.1: <bound method> = bound_method %int_9, %impl.elem0.loc8_30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.268]
// CHECK:STDOUT:   %specific_fn.loc8_30: <specific function> = specific_function %impl.elem0.loc8_30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_30.2: <bound method> = bound_method %int_9, %specific_fn.loc8_30 [concrete = constants.%bound_method.e3f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_30: init %i32 = call %bound_method.loc8_30.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc8_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_30 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc8_30.2: %i32 = converted %int_9, %.loc8_30.1 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %addr.loc8: %ptr.d9e = addr_of %.loc8_31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%.loc8_27.2, %.loc8_30.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_31.2: init %C = in_place_init %C__carbon_thunk.call.loc8, %.loc8_31.1
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: ref %C = temporary %.loc8_31.1, %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %C = acquire_value %.loc8_31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc9_28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_27.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e6]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_27.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.0d0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_27.2(%int_8.loc9) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc9_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc9_27.2: %i32 = converted %int_8.loc9, %.loc9_27.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr.loc9: %ptr.d9e = addr_of %.loc9_28.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc9_27.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_28.2: init %C = in_place_init %C__carbon_thunk.call.loc9, %.loc9_28.1
// CHECK:STDOUT:   %.loc9_14: type = splice_block %C.ref.loc9_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_28.3: ref %C = temporary %.loc9_28.1, %.loc9_28.2
// CHECK:STDOUT:   %.loc9_28.4: %C = acquire_value %.loc9_28.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_28.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc17: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17: %.edb = impl_witness_access constants.%As.impl_witness.bf0, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.035]
// CHECK:STDOUT:   %bound_method.loc17_21.1: <bound method> = bound_method %int_8.loc17, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_21.2: <bound method> = bound_method %int_8.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.493]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc17_21.2(%int_8.loc17) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc17_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc17_21.2: %i32 = converted %int_8.loc17, %.loc17_21.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc17_14: type = splice_block %C.ref.loc17 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc17: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_21.3: %C = converted %.loc17_21.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %c3: %C = value_binding c3, <error> [concrete = <error>]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_28.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9_28: <bound method> = bound_method %.loc9_28.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9_28(%.loc9_28.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_31.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f77
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_31: <bound method> = bound_method %.loc8_31.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_31(%.loc8_31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
