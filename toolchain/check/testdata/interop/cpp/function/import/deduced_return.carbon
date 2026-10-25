// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/deduced_return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/deduced_return.carbon

// --- scalar_and_reference.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> auto Scalar(T value) { return value; }
template<typename T> decltype(auto) Identity(T value) { return value; }
template<typename T> auto& AutoRef(T& value) { return value; }
template<typename T> decltype(auto) Ref(T& value) { return (value); }
''';

fn Call() {
  var value: i32 = 1;
  var wide: i64 = 2;
  //@dump-sem-ir-begin
  Cpp.Ref(ref value) = 42;
  Cpp.Ref(i64, ref wide) = 43;
  Cpp.AutoRef(ref value) = 44;
  let unused identity: i32 = Cpp.Identity(value);
  let unused deduced: i32 = Cpp.Scalar(value);
  let unused explicit: i64 = Cpp.Scalar(i64, wide);
  //@dump-sem-ir-end
}

// --- redeclaration.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> auto Redeclared(T);
template<typename T> auto Redeclared(T value) { return value; }
''';

fn Call(value: i32) -> i32 {
  //@dump-sem-ir-begin
  return Cpp.Redeclared(value);
  //@dump-sem-ir-end
}

// --- non_template_member.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct Value {
  int value;
  auto Get() const { return value; }
};
''';

fn Call(value: Cpp.Value) -> i32 {
  //@dump-sem-ir-begin
  return value.Get();
  //@dump-sem-ir-end
}

// --- forwarding_member.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> struct Holder {
  T value;
  template<typename... Args> decltype(auto) Set(Args&&... args) {
    value = T(static_cast<Args&&>(args)...);
    return (value);
  }
};
''';

fn Call() {
  var holder: Cpp.Holder(i32) = Cpp.Holder(i32).Holder();
  var value: i32 = 1;
  //@dump-sem-ir-begin
  holder.Set(ref value) = 42;
  holder.Set(i32, 43) = 44;
  //@dump-sem-ir-end
}

// --- void.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> decltype(auto) Discard(T& value) { value = 0; }
struct Empty {};
template<typename T> struct Holder;
template<> struct Holder<Empty> {
  template<typename... Args> decltype(auto) Set(Args&&...) {}
};
''';

fn Call() {
  var value: i32 = 1;
  var holder: Cpp.Holder(Cpp.Empty) = Cpp.Holder(Cpp.Empty).Holder();
  //@dump-sem-ir-begin
  Cpp.Discard(ref value);
  holder.Set();
  holder.Set(ref value);
  //@dump-sem-ir-end
}

// --- returned_class.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> struct Result {
  T first;
  T second;
  T third;
  T fourth;
  T fifth;
};
template<typename T> auto MakeResult(T value) {
  return Result<T>{value, value + 1, value + 2, value + 3, value + 4};
}
''';

// Import the result's class through return deduction, not an explicit type use.
// Five i32 fields keep the indirect return without platform-specific i64 mapping.
fn Call(value: i32) -> i32 {
  //@dump-sem-ir-begin
  return Cpp.MakeResult(value).fifth;
  //@dump-sem-ir-end
}

// --- unselected_body.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> auto Select(T value) {
  static_assert(sizeof(T) == 0, "unselected body instantiated");
  return value.missing();
}
auto Select(int value) { return value; }
''';

fn Call(value: i32) -> i32 {
  //@dump-sem-ir-begin
  return Cpp.Select(value);
  //@dump-sem-ir-end
}

// --- fail_undefined_template.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> auto UndefinedTemplate(T);
''';

fn Call(value: i32) {
  // CHECK:STDERR: fail_undefined_template.carbon:[[@LINE+3]]:3: error: function 'UndefinedTemplate<int>' with deduced return type cannot be used before it is defined [CppInteropParseError]
  // CHECK:STDERR: fail_undefined_template.carbon:[[@LINE-5]]:27: note: 'UndefinedTemplate<int>' declared here [CppInteropParseNote]
  // CHECK:STDERR:
  Cpp.UndefinedTemplate(value);
}

// --- fail_invalid_return.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
// CHECK:STDERR: fail_invalid_return.carbon:[[@LINE+1]]:64: error: member reference base type 'int' is not a structure or union [CppInteropParseError]
template<typename T> auto InvalidReturn(T value) { return value.missing(); }
''';

fn Call(value: i32) {
  // CHECK:STDERR: fail_invalid_return.carbon:[[@LINE+2]]:3: note: in instantiation of function template specialization 'InvalidReturn<int>' requested here [CppInteropParseNote]
  // CHECK:STDERR:
  Cpp.InvalidReturn(value);
}

// --- fail_static_assert.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> auto Rejected(T value) {
  // CHECK:STDERR: fail_static_assert.carbon:[[@LINE+1]]:17-30: error: static assertion failed due to requirement 'sizeof(int) == 0': return deduction rejected [CppInteropParseError]
  static_assert(sizeof(T) == 0, "return deduction rejected");
  return value;
}
''';

fn Call(value: i32) {
  // CHECK:STDERR: fail_static_assert.carbon:[[@LINE+3]]:3: note: in instantiation of function template specialization 'Rejected<int>' requested here [CppInteropParseNote]
  // CHECK:STDERR: fail_static_assert.carbon:[[@LINE-7]]:17-30: note: expression evaluates to '4 == 0' [CppInteropParseNote]
  // CHECK:STDERR:
  Cpp.Rejected(value);
  // Recovery must handle a specialization whose return type was deduced before
  // its body produced an error.
  Cpp.Rejected(value);
}

// CHECK:STDOUT: --- scalar_and_reference.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.c0e: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e56: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b1b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8f5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.dcd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8f5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.246: %ImplicitAs.type.e56 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.b1b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4cd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.246) [concrete]
// CHECK:STDOUT:   %.7ee: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4cd, %ImplicitAs.facet.246 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.099: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.dcd, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %Ref.cpp_overload_set.type: type = cpp_overload_set_type @Ref.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Ref.cpp_overload_set.value: %Ref.cpp_overload_set.type = cpp_overload_set_value @Ref.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Ref.type.7b771e.1: type = fn_type @Ref.1 [concrete]
// CHECK:STDOUT:   %Ref.eb051e.1: %Ref.type.7b771e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.98e: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8 [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Ref.type.7b771e.2: type = fn_type @Ref.2 [concrete]
// CHECK:STDOUT:   %Ref.eb051e.2: %Ref.type.7b771e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_43.1d4: Core.IntLiteral = int_value 43 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.39c: <bound method> = bound_method %int_43.1d4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.dcd [concrete]
// CHECK:STDOUT:   %bound_method.166: <bound method> = bound_method %int_43.1d4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.099 [concrete]
// CHECK:STDOUT:   %int_43.482: %i64 = int_value 43 [concrete]
// CHECK:STDOUT:   %AutoRef.cpp_overload_set.type: type = cpp_overload_set_type @AutoRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %AutoRef.cpp_overload_set.value: %AutoRef.cpp_overload_set.type = cpp_overload_set_value @AutoRef.cpp_overload_set [concrete]
// CHECK:STDOUT:   %AutoRef.type: type = fn_type @AutoRef [concrete]
// CHECK:STDOUT:   %AutoRef: %AutoRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_44.435: Core.IntLiteral = int_value 44 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.59e: <bound method> = bound_method %int_44.435, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.0fa: <bound method> = bound_method %int_44.435, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8 [concrete]
// CHECK:STDOUT:   %int_44.584: %i32 = int_value 44 [concrete]
// CHECK:STDOUT:   %identity.patt: %pattern_type.d94 = value_binding_pattern identity [concrete]
// CHECK:STDOUT:   %Identity.cpp_overload_set.type: type = cpp_overload_set_type @Identity.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Identity.cpp_overload_set.value: %Identity.cpp_overload_set.type = cpp_overload_set_value @Identity.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Identity.type: type = fn_type @Identity [concrete]
// CHECK:STDOUT:   %Identity: %Identity.type = struct_value () [concrete]
// CHECK:STDOUT:   %deduced.patt: %pattern_type.d94 = value_binding_pattern deduced [concrete]
// CHECK:STDOUT:   %Scalar.cpp_overload_set.type: type = cpp_overload_set_type @Scalar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Scalar.cpp_overload_set.value: %Scalar.cpp_overload_set.type = cpp_overload_set_value @Scalar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Scalar.type.21b146.1: type = fn_type @Scalar.1 [concrete]
// CHECK:STDOUT:   %Scalar.a86798.1: %Scalar.type.21b146.1 = struct_value () [concrete]
// CHECK:STDOUT:   %explicit.patt: %pattern_type.c0e = value_binding_pattern explicit [concrete]
// CHECK:STDOUT:   %Scalar.type.21b146.2: type = fn_type @Scalar.2 [concrete]
// CHECK:STDOUT:   %Scalar.a86798.2: %Scalar.type.21b146.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Ref = %Ref.cpp_overload_set.value
// CHECK:STDOUT:     .AutoRef = %AutoRef.cpp_overload_set.value
// CHECK:STDOUT:     .Identity = %Identity.cpp_overload_set.value
// CHECK:STDOUT:     .Scalar = %Scalar.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT:   %Ref.cpp_overload_set.value: %Ref.cpp_overload_set.type = cpp_overload_set_value @Ref.cpp_overload_set [concrete = constants.%Ref.cpp_overload_set.value]
// CHECK:STDOUT:   %Ref.decl.a0d147.1: %Ref.type.7b771e.1 = fn_decl @Ref.1 [concrete = constants.%Ref.eb051e.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ref.decl.a0d147.2: %Ref.type.7b771e.2 = fn_decl @Ref.2 [concrete = constants.%Ref.eb051e.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AutoRef.cpp_overload_set.value: %AutoRef.cpp_overload_set.type = cpp_overload_set_value @AutoRef.cpp_overload_set [concrete = constants.%AutoRef.cpp_overload_set.value]
// CHECK:STDOUT:   %AutoRef.decl: %AutoRef.type = fn_decl @AutoRef [concrete = constants.%AutoRef] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Identity.cpp_overload_set.value: %Identity.cpp_overload_set.type = cpp_overload_set_value @Identity.cpp_overload_set [concrete = constants.%Identity.cpp_overload_set.value]
// CHECK:STDOUT:   %Identity.decl: %Identity.type = fn_decl @Identity [concrete = constants.%Identity] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Scalar.cpp_overload_set.value: %Scalar.cpp_overload_set.type = cpp_overload_set_value @Scalar.cpp_overload_set [concrete = constants.%Scalar.cpp_overload_set.value]
// CHECK:STDOUT:   %Scalar.decl.3e8295.1: %Scalar.type.21b146.1 = fn_decl @Scalar.1 [concrete = constants.%Scalar.a86798.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Scalar.decl.3e8295.2: %Scalar.type.21b146.2 = fn_decl @Scalar.2 [concrete = constants.%Scalar.a86798.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Ref.ref.loc15: %Ref.cpp_overload_set.type = name_ref Ref, imports.%Ref.cpp_overload_set.value [concrete = constants.%Ref.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref.loc15: ref %i32 = name_ref value, %value
// CHECK:STDOUT:   %.loc15_11: %i32 = ref_tag %value.ref.loc15
// CHECK:STDOUT:   %Ref.call.loc15: ref %i32 = call imports.%Ref.decl.a0d147.1(%value.ref.loc15)
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc15: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc15_22.1: <bound method> = bound_method %int_42, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8]
// CHECK:STDOUT:   %bound_method.loc15_22.2: <bound method> = bound_method %int_42, %specific_fn.loc15 [concrete = constants.%bound_method.98e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i32 = call %bound_method.loc15_22.2(%int_42) [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc15_22: init %i32 = converted %int_42, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   assign %Ref.call.loc15, %.loc15_22
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Ref.ref.loc16: %Ref.cpp_overload_set.type = name_ref Ref, imports.%Ref.cpp_overload_set.value [concrete = constants.%Ref.cpp_overload_set.value]
// CHECK:STDOUT:   %i64.loc16: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %wide.ref.loc16: ref %i64 = name_ref wide, %wide
// CHECK:STDOUT:   %.loc16_16: %i64 = ref_tag %wide.ref.loc16
// CHECK:STDOUT:   %Ref.call.loc16: ref %i64 = call imports.%Ref.decl.a0d147.2(%wide.ref.loc16)
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete = constants.%int_43.1d4]
// CHECK:STDOUT:   %impl.elem0.loc16: %.7ee = impl_witness_access constants.%ImplicitAs.impl_witness.b1b, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.dcd]
// CHECK:STDOUT:   %bound_method.loc16_26.1: <bound method> = bound_method %int_43, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.39c]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.099]
// CHECK:STDOUT:   %bound_method.loc16_26.2: <bound method> = bound_method %int_43, %specific_fn.loc16 [concrete = constants.%bound_method.166]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %i64 = call %bound_method.loc16_26.2(%int_43) [concrete = constants.%int_43.482]
// CHECK:STDOUT:   %.loc16_26: init %i64 = converted %int_43, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_43.482]
// CHECK:STDOUT:   assign %Ref.call.loc16, %.loc16_26
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %AutoRef.ref: %AutoRef.cpp_overload_set.type = name_ref AutoRef, imports.%AutoRef.cpp_overload_set.value [concrete = constants.%AutoRef.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref.loc17: ref %i32 = name_ref value, %value
// CHECK:STDOUT:   %.loc17_15: %i32 = ref_tag %value.ref.loc17
// CHECK:STDOUT:   %AutoRef.call: ref %i32 = call imports.%AutoRef.decl(%value.ref.loc17)
// CHECK:STDOUT:   %int_44: Core.IntLiteral = int_value 44 [concrete = constants.%int_44.435]
// CHECK:STDOUT:   %impl.elem0.loc17: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc17_26.1: <bound method> = bound_method %int_44, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.59e]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8]
// CHECK:STDOUT:   %bound_method.loc17_26.2: <bound method> = bound_method %int_44, %specific_fn.loc17 [concrete = constants.%bound_method.0fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %i32 = call %bound_method.loc17_26.2(%int_44) [concrete = constants.%int_44.584]
// CHECK:STDOUT:   %.loc17_26: init %i32 = converted %int_44, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_44.584]
// CHECK:STDOUT:   assign %AutoRef.call, %.loc17_26
// CHECK:STDOUT:   %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Identity.ref: %Identity.cpp_overload_set.type = name_ref Identity, imports.%Identity.cpp_overload_set.value [concrete = constants.%Identity.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref.loc18: ref %i32 = name_ref value, %value
// CHECK:STDOUT:   %.loc18_43: %i32 = acquire_value %value.ref.loc18
// CHECK:STDOUT:   %Identity.call: init %i32 = call imports.%Identity.decl(%.loc18_43)
// CHECK:STDOUT:   %.loc18_48.1: %i32 = value_of_initializer %Identity.call
// CHECK:STDOUT:   %.loc18_48.2: %i32 = converted %Identity.call, %.loc18_48.1
// CHECK:STDOUT:   %i32.loc18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %identity: %i32 = wrapper_binding identity, %.loc18_48.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %identity.patt: %pattern_type.d94 = value_binding_pattern identity [concrete = constants.%identity.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Scalar.ref.loc19: %Scalar.cpp_overload_set.type = name_ref Scalar, imports.%Scalar.cpp_overload_set.value [concrete = constants.%Scalar.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref.loc19: ref %i32 = name_ref value, %value
// CHECK:STDOUT:   %.loc19_40: %i32 = acquire_value %value.ref.loc19
// CHECK:STDOUT:   %Scalar.call.loc19: init %i32 = call imports.%Scalar.decl.3e8295.1(%.loc19_40)
// CHECK:STDOUT:   %.loc19_45.1: %i32 = value_of_initializer %Scalar.call.loc19
// CHECK:STDOUT:   %.loc19_45.2: %i32 = converted %Scalar.call.loc19, %.loc19_45.1
// CHECK:STDOUT:   %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %deduced: %i32 = wrapper_binding deduced, %.loc19_45.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %deduced.patt: %pattern_type.d94 = value_binding_pattern deduced [concrete = constants.%deduced.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Scalar.ref.loc20: %Scalar.cpp_overload_set.type = name_ref Scalar, imports.%Scalar.cpp_overload_set.value [concrete = constants.%Scalar.cpp_overload_set.value]
// CHECK:STDOUT:   %i64.loc20_41: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %wide.ref.loc20: ref %i64 = name_ref wide, %wide
// CHECK:STDOUT:   %.loc20_46: %i64 = acquire_value %wide.ref.loc20
// CHECK:STDOUT:   %Scalar.call.loc20: init %i64 = call imports.%Scalar.decl.3e8295.2(%.loc20_46)
// CHECK:STDOUT:   %.loc20_50.1: %i64 = value_of_initializer %Scalar.call.loc20
// CHECK:STDOUT:   %.loc20_50.2: %i64 = converted %Scalar.call.loc20, %.loc20_50.1
// CHECK:STDOUT:   %i64.loc20_24: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %explicit: %i64 = wrapper_binding explicit, %.loc20_50.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit.patt: %pattern_type.c0e = value_binding_pattern explicit [concrete = constants.%explicit.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Redeclared.cpp_overload_set.type: type = cpp_overload_set_type @Redeclared.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Redeclared.cpp_overload_set.value: %Redeclared.cpp_overload_set.type = cpp_overload_set_value @Redeclared.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Redeclared.type: type = fn_type @Redeclared [concrete]
// CHECK:STDOUT:   %Redeclared: %Redeclared.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Redeclared = %Redeclared.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Redeclared.cpp_overload_set.value: %Redeclared.cpp_overload_set.type = cpp_overload_set_value @Redeclared.cpp_overload_set [concrete = constants.%Redeclared.cpp_overload_set.value]
// CHECK:STDOUT:   %Redeclared.decl: %Redeclared.type = fn_decl @Redeclared [concrete = constants.%Redeclared] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%value.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Redeclared.ref: %Redeclared.cpp_overload_set.type = name_ref Redeclared, imports.%Redeclared.cpp_overload_set.value [concrete = constants.%Redeclared.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref: %i32 = name_ref value, %value
// CHECK:STDOUT:   %Redeclared.call: init %i32 = call imports.%Redeclared.decl(%value.ref)
// CHECK:STDOUT:   return %Redeclared.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_template_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Value: type = class_type @Value [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Value.Get.cpp_overload_set.type: type = cpp_overload_set_type @Value.Get.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Value.Get.cpp_overload_set.value: %Value.Get.cpp_overload_set.type = cpp_overload_set_value @Value.Get.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Get__carbon_thunk.type: type = fn_type @Get__carbon_thunk [concrete]
// CHECK:STDOUT:   %Get__carbon_thunk: %Get__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Value.Get.cpp_overload_set.value: %Value.Get.cpp_overload_set.type = cpp_overload_set_value @Value.Get.cpp_overload_set [concrete = constants.%Value.Get.cpp_overload_set.value]
// CHECK:STDOUT:   %Get__carbon_thunk.decl: %Get__carbon_thunk.type = fn_decl @Get__carbon_thunk [concrete = constants.%Get__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%value.param: %Value) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %Value = name_ref value, %value
// CHECK:STDOUT:   %Get.ref: %Value.Get.cpp_overload_set.type = name_ref Get, imports.%Value.Get.cpp_overload_set.value [concrete = constants.%Value.Get.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %value.ref, %Get.ref
// CHECK:STDOUT:   %Get__carbon_thunk.call: init %i32 = call imports.%Get__carbon_thunk.decl(%value.ref)
// CHECK:STDOUT:   return %Get__carbon_thunk.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- forwarding_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Holder: type = class_type @Holder [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %Holder.Set.cpp_overload_set.type: type = cpp_overload_set_type @Holder.Set.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Holder.Set.cpp_overload_set.value: %Holder.Set.cpp_overload_set.type = cpp_overload_set_value @Holder.Set.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Holder.Set.type.8a4d0e.1: type = fn_type @Holder.Set.1 [concrete]
// CHECK:STDOUT:   %Holder.Set.a7c0a2.1: %Holder.Set.type.8a4d0e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.98e: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %int_43.1d4: Core.IntLiteral = int_value 43 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Set__carbon_thunk.type: type = fn_type @Set__carbon_thunk [concrete]
// CHECK:STDOUT:   %Set__carbon_thunk: %Set__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_44.435: Core.IntLiteral = int_value 44 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.59e: <bound method> = bound_method %int_44.435, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.0fa: <bound method> = bound_method %int_44.435, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_44.584: %i32 = int_value 44 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT:   %Holder.Set.cpp_overload_set.value: %Holder.Set.cpp_overload_set.type = cpp_overload_set_value @Holder.Set.cpp_overload_set [concrete = constants.%Holder.Set.cpp_overload_set.value]
// CHECK:STDOUT:   %Holder.Set.decl.cd89ba.1: %Holder.Set.type.8a4d0e.1 = fn_decl @Holder.Set.1 [concrete = constants.%Holder.Set.a7c0a2.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Set__carbon_thunk.decl: %Set__carbon_thunk.type = fn_decl @Set__carbon_thunk [concrete = constants.%Set__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %holder.ref.loc18: ref %Holder = name_ref holder, %holder
// CHECK:STDOUT:   %Set.ref.loc18: %Holder.Set.cpp_overload_set.type = name_ref Set, imports.%Holder.Set.cpp_overload_set.value [concrete = constants.%Holder.Set.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc18_9: <bound method> = bound_method %holder.ref.loc18, %Set.ref.loc18
// CHECK:STDOUT:   %value.ref: ref %i32 = name_ref value, %value
// CHECK:STDOUT:   %.loc18_14: %i32 = ref_tag %value.ref
// CHECK:STDOUT:   %Holder.Set.call: ref %i32 = call imports.%Holder.Set.decl.cd89ba.1(%holder.ref.loc18, %value.ref)
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc18: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc18_25.1: <bound method> = bound_method %int_42, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_25.2: <bound method> = bound_method %int_42, %specific_fn.loc18 [concrete = constants.%bound_method.98e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %i32 = call %bound_method.loc18_25.2(%int_42) [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc18_25: init %i32 = converted %int_42, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   assign %Holder.Set.call, %.loc18_25
// CHECK:STDOUT:   %holder.ref.loc19: ref %Holder = name_ref holder, %holder
// CHECK:STDOUT:   %Set.ref.loc19: %Holder.Set.cpp_overload_set.type = name_ref Set, imports.%Holder.Set.cpp_overload_set.value [concrete = constants.%Holder.Set.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc19_9: <bound method> = bound_method %holder.ref.loc19, %Set.ref.loc19
// CHECK:STDOUT:   %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete = constants.%int_43.1d4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr.loc19: %ptr.a17 = addr_of %args.var
// CHECK:STDOUT:   %Set__carbon_thunk.call: ref %i32 = call imports.%Set__carbon_thunk.decl(%holder.ref.loc19, %addr.loc19)
// CHECK:STDOUT:   %int_44: Core.IntLiteral = int_value 44 [concrete = constants.%int_44.435]
// CHECK:STDOUT:   %impl.elem0.loc19: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc19_23.1: <bound method> = bound_method %int_44, %impl.elem0.loc19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.59e]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_23.2: <bound method> = bound_method %int_44, %specific_fn.loc19 [concrete = constants.%bound_method.0fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19: init %i32 = call %bound_method.loc19_23.2(%int_44) [concrete = constants.%int_44.584]
// CHECK:STDOUT:   %.loc19: init %i32 = converted %int_44, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19 [concrete = constants.%int_44.584]
// CHECK:STDOUT:   assign %Set__carbon_thunk.call, %.loc19
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- void.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Holder.type: type = cpp_type_template_type Holder [concrete]
// CHECK:STDOUT:   %Holder.template: %Holder.type = struct_value () [concrete]
// CHECK:STDOUT:   %Empty: type = class_type @Empty [concrete]
// CHECK:STDOUT:   %Holder: type = class_type @Holder [concrete]
// CHECK:STDOUT:   %Discard.cpp_overload_set.type: type = cpp_overload_set_type @Discard.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Discard.cpp_overload_set.value: %Discard.cpp_overload_set.type = cpp_overload_set_value @Discard.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Discard.type: type = fn_type @Discard [concrete]
// CHECK:STDOUT:   %Discard: %Discard.type = struct_value () [concrete]
// CHECK:STDOUT:   %Holder.Set.cpp_overload_set.type: type = cpp_overload_set_type @Holder.Set.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Holder.Set.cpp_overload_set.value: %Holder.Set.cpp_overload_set.type = cpp_overload_set_value @Holder.Set.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Holder.Set.type.8a4d0e.1: type = fn_type @Holder.Set.1 [concrete]
// CHECK:STDOUT:   %Holder.Set.a7c0a2.1: %Holder.Set.type.8a4d0e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Holder.Set.type.8a4d0e.2: type = fn_type @Holder.Set.2 [concrete]
// CHECK:STDOUT:   %Holder.Set.a7c0a2.2: %Holder.Set.type.8a4d0e.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Holder = %Holder.template
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Discard = %Discard.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Holder.template: %Holder.type = struct_value () [concrete = constants.%Holder.template]
// CHECK:STDOUT:   %Empty.decl: type = class_decl @Empty [concrete = constants.%Empty] {} {}
// CHECK:STDOUT:   %Discard.cpp_overload_set.value: %Discard.cpp_overload_set.type = cpp_overload_set_value @Discard.cpp_overload_set [concrete = constants.%Discard.cpp_overload_set.value]
// CHECK:STDOUT:   %Discard.decl: %Discard.type = fn_decl @Discard [concrete = constants.%Discard] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Holder.Set.cpp_overload_set.value: %Holder.Set.cpp_overload_set.type = cpp_overload_set_value @Holder.Set.cpp_overload_set [concrete = constants.%Holder.Set.cpp_overload_set.value]
// CHECK:STDOUT:   %Holder.Set.decl.cd89ba.1: %Holder.Set.type.8a4d0e.1 = fn_decl @Holder.Set.1 [concrete = constants.%Holder.Set.a7c0a2.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Holder.Set.decl.cd89ba.2: %Holder.Set.type.8a4d0e.2 = fn_decl @Holder.Set.2 [concrete = constants.%Holder.Set.a7c0a2.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Discard.ref: %Discard.cpp_overload_set.type = name_ref Discard, imports.%Discard.cpp_overload_set.value [concrete = constants.%Discard.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref.loc17: ref %i32 = name_ref value, %value
// CHECK:STDOUT:   %.loc17: %i32 = ref_tag %value.ref.loc17
// CHECK:STDOUT:   %Discard.call: init %empty_tuple.type = call imports.%Discard.decl(%value.ref.loc17)
// CHECK:STDOUT:   %holder.ref.loc18: ref %Holder = name_ref holder, %holder
// CHECK:STDOUT:   %Set.ref.loc18: %Holder.Set.cpp_overload_set.type = name_ref Set, imports.%Holder.Set.cpp_overload_set.value [concrete = constants.%Holder.Set.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %holder.ref.loc18, %Set.ref.loc18
// CHECK:STDOUT:   %Holder.Set.call.loc18: init %empty_tuple.type = call imports.%Holder.Set.decl.cd89ba.1(%holder.ref.loc18)
// CHECK:STDOUT:   %holder.ref.loc19: ref %Holder = name_ref holder, %holder
// CHECK:STDOUT:   %Set.ref.loc19: %Holder.Set.cpp_overload_set.type = name_ref Set, imports.%Holder.Set.cpp_overload_set.value [concrete = constants.%Holder.Set.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %holder.ref.loc19, %Set.ref.loc19
// CHECK:STDOUT:   %value.ref.loc19: ref %i32 = name_ref value, %value
// CHECK:STDOUT:   %.loc19: %i32 = ref_tag %value.ref.loc19
// CHECK:STDOUT:   %Holder.Set.call.loc19: init %empty_tuple.type = call imports.%Holder.Set.decl.cd89ba.2(%holder.ref.loc19, %value.ref.loc19)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- returned_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %MakeResult.cpp_overload_set.type: type = cpp_overload_set_type @MakeResult.cpp_overload_set [concrete]
// CHECK:STDOUT:   %MakeResult.cpp_overload_set.value: %MakeResult.cpp_overload_set.type = cpp_overload_set_value @MakeResult.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Result: type = class_type @Result [concrete]
// CHECK:STDOUT:   %pattern_type.36a: type = pattern_type %Result [concrete]
// CHECK:STDOUT:   %ptr.bd7: type = ptr_type %Result [concrete]
// CHECK:STDOUT:   %MakeResult__carbon_thunk.type: type = fn_type @MakeResult__carbon_thunk [concrete]
// CHECK:STDOUT:   %MakeResult__carbon_thunk: %MakeResult__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Result.elem: type = unbound_element_type %Result, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %self.param_patt.ef7: %pattern_type.36a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.ca3: %pattern_type.36a = wrapper_binding_pattern self, %self.param_patt.ef7 [concrete]
// CHECK:STDOUT:   %Result.cpp_destructor.type: type = fn_type @Result.cpp_destructor [concrete]
// CHECK:STDOUT:   %Result.cpp_destructor: %Result.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc21 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc21 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MakeResult = %MakeResult.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeResult.cpp_overload_set.value: %MakeResult.cpp_overload_set.type = cpp_overload_set_value @MakeResult.cpp_overload_set [concrete = constants.%MakeResult.cpp_overload_set.value]
// CHECK:STDOUT:   %MakeResult__carbon_thunk.decl: %MakeResult__carbon_thunk.type = fn_decl @MakeResult__carbon_thunk [concrete = constants.%MakeResult__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Result.cpp_destructor.decl: %Result.cpp_destructor.type = fn_decl @Result.cpp_destructor [concrete = constants.%Result.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc21 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.36a = ref_param_pattern [concrete = constants.%self.param_patt.ef7]
// CHECK:STDOUT:     %self.patt: %pattern_type.36a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ca3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Result = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Result = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%value.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MakeResult.ref: %MakeResult.cpp_overload_set.type = name_ref MakeResult, imports.%MakeResult.cpp_overload_set.value [concrete = constants.%MakeResult.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref: %i32 = name_ref value, %value
// CHECK:STDOUT:   %.loc21_30.1: ref %Result = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.bd7 = addr_of %.loc21_30.1
// CHECK:STDOUT:   %MakeResult__carbon_thunk.call: init %empty_tuple.type = call imports.%MakeResult__carbon_thunk.decl(%value.ref, %addr)
// CHECK:STDOUT:   %.loc21_30.2: init %Result to %.loc21_30.1 = mark_in_place_init %MakeResult__carbon_thunk.call
// CHECK:STDOUT:   %.loc21_30.3: ref %Result = temporary %.loc21_30.1, %.loc21_30.2
// CHECK:STDOUT:   %fifth.ref: %Result.elem = name_ref fifth, @Result.%field_decl.5 [concrete = @Result.%field_decl.5]
// CHECK:STDOUT:   %.loc21_31.1: ref %i32 = class_element_access %.loc21_30.3, element4
// CHECK:STDOUT:   %.loc21_31.2: %i32 = acquire_value %.loc21_31.1
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc21_31.1: <bound method> = bound_method %.loc21_31.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_31.2: <bound method> = bound_method %.loc21_31.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc21_31.2(%.loc21_31.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc21_30.3, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc21_30.3)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc21(%self.param: ref %Result) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc21(%self.param: ref %Result) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Result.cpp_destructor.call: init %empty_tuple.type = call imports.%Result.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unselected_body.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Select.cpp_overload_set.type: type = cpp_overload_set_type @Select.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Select.cpp_overload_set.value: %Select.cpp_overload_set.type = cpp_overload_set_value @Select.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Select.type: type = fn_type @Select [concrete]
// CHECK:STDOUT:   %Select: %Select.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Select = %Select.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Select.cpp_overload_set.value: %Select.cpp_overload_set.type = cpp_overload_set_value @Select.cpp_overload_set [concrete = constants.%Select.cpp_overload_set.value]
// CHECK:STDOUT:   %Select.decl: %Select.type = fn_decl @Select [concrete = constants.%Select] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%value.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Select.ref: %Select.cpp_overload_set.type = name_ref Select, imports.%Select.cpp_overload_set.value [concrete = constants.%Select.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref: %i32 = name_ref value, %value
// CHECK:STDOUT:   %Select.call: init %i32 = call imports.%Select.decl(%value.ref)
// CHECK:STDOUT:   return %Select.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
