// RUN: mlir-opt -test-patterns -mlir-print-debuginfo -mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: verifyFusedLocs
func.func @verifyFusedLocs(%arg0 : i32) -> i32 {
  %0 = "test.op_a"(%arg0) {attr = 10 : i32} : (i32) -> i32 loc("a")
  %result = "test.op_a"(%0) {attr = 20 : i32} : (i32) -> i32 loc("b")

  // CHECK: "test.op_b"(%arg0) {attr = 10 : i32} : (i32) -> i32 loc("a")
  // CHECK: "test.op_b"(%arg0) {attr = 20 : i32} : (i32) -> i32 loc(fused["b", "a"])
  return %result : i32
}

// CHECK-LABEL: verifyDesignatedLoc
func.func @verifyDesignatedLoc(%arg0 : i32) -> i32 {
  %0 = "test.loc_src"(%arg0) : (i32) -> i32 loc("loc3")
  %1 = "test.loc_src"(%0) : (i32) -> i32 loc("loc2")
  %2 = "test.loc_src"(%1) : (i32) -> i32 loc("loc1")
  "test.loc_src_no_res"(%2) : (i32) -> () loc("loc4")

  // CHECK: "test.loc_dst"({{.*}}) : (i32) -> i32 loc("loc1")
  // CHECK: "test.loc_dst"({{.*}}) : (i32) -> i32 loc("named")
  // CHECK: "test.loc_dst"({{.*}}) : (i32) -> i32 loc(fused<"fused">["loc2", "loc3"])
  // CHECK: "test.loc_dst_no_res"({{.*}}) : (i32) -> () loc("loc4")
  return %1 : i32
}

// CHECK-LABEL: verifyZeroResult
func.func @verifyZeroResult(%arg0 : i32) {
  // CHECK: "test.op_i"(%arg0) : (i32) -> ()
  "test.op_h"(%arg0) : (i32) -> ()
  return
}

// CHECK-LABEL: verifyZeroArg
func.func @verifyZeroArg() -> i32 {
  // CHECK: "test.op_k"() : () -> i32
  %0 = "test.op_j"() : () -> i32
  return %0 : i32
}

// CHECK-LABEL: testIgnoreArgMatch
// CHECK-SAME: (%{{[a-z0-9]*}}: i32 loc({{[^)]*}}), %[[ARG1:[a-z0-9]*]]: i32 loc({{[^)]*}}),
func.func @testIgnoreArgMatch(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f32) {
  // CHECK: "test.ignore_arg_match_dst"(%[[ARG1]]) {f = 15 : i64}
  "test.ignore_arg_match_src"(%arg0, %arg1, %arg2) {d = 42, e = 24, f = 15} : (i32, i32, i32) -> ()

  // CHECK: test.ignore_arg_match_src
  // Not match because wrong type for $c.
  "test.ignore_arg_match_src"(%arg0, %arg1, %arg3) {d = 42, e = 24, f = 15} : (i32, i32, f32) -> ()

  // CHECK: test.ignore_arg_match_src
  // Not match because wrong type for $f.
  "test.ignore_arg_match_src"(%arg0, %arg1, %arg2) {d = 42 : i32, e = 24, f = 15} : (i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: verifyInterleavedOperandAttribute
// CHECK-SAME:    %[[ARG0:.*]]: i32 loc({{[^)]*}}), %[[ARG1:.*]]: i32 loc({{[^)]*}})
func.func @verifyInterleavedOperandAttribute(%arg0: i32, %arg1: i32) {
  // CHECK: "test.interleaved_operand_attr2"(%[[ARG0]], %[[ARG1]]) {attr1 = 15 : i64, attr2 = 42 : i64}
  "test.interleaved_operand_attr1"(%arg0, %arg1) {attr1 = 15, attr2 = 42} : (i32, i32) -> ()
  return
}

// CHECK-LABEL: verifyBenefit
func.func @verifyBenefit(%arg0 : i32) -> i32 {
  %0 = "test.op_d"(%arg0) : (i32) -> i32
  %1 = "test.op_g"(%arg0) : (i32) -> i32
  %2 = "test.op_g"(%1) : (i32) -> i32

  // CHECK: "test.op_f"(%arg0)
  // CHECK: "test.op_b"(%arg0) {attr = 34 : i32}
  return %0 : i32
}

// CHECK-LABEL: verifyNativeCodeCall
func.func @verifyNativeCodeCall(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK: %0 = "test.native_code_call2"(%arg0) {attr = [42, 24]} : (i32) -> i32
  // CHECK:  return %0, %arg1
  %0 = "test.native_code_call1"(%arg0, %arg1) {choice = true, attr1 = 42, attr2 = 24} : (i32, i32) -> (i32)
  %1 = "test.native_code_call1"(%arg0, %arg1) {choice = false, attr1 = 42, attr2 = 24} : (i32, i32) -> (i32)
  return %0, %1: i32, i32
}

// CHECK-LABEL: verifyAuxiliaryNativeCodeCall
func.func @verifyAuxiliaryNativeCodeCall(%arg0: i32) -> (i32) {
  // CHECK: test.op_i
  // CHECK: test.op_k
  %0 = "test.native_code_call3"(%arg0) : (i32) -> (i32)
  return %0 : i32
}

// CHECK-LABEL: verifyNativeCodeCallBinding
func.func @verifyNativeCodeCallBinding(%arg0 : i32) -> (i32) {
  %0 = "test.op_k"() : () -> (i32)
  // CHECK: %[[A:.*]], %[[B:.*]] = "test.native_code_call5"(%1, %1) : (i32, i32) -> (i32, i32)
  %1, %2 = "test.native_code_call4"(%0) : (i32) -> (i32, i32)
  %3 = "test.constant"() {value = 1 : i8} : () -> i8
  // %3 is i8 so it'll fail at GetFirstI32Result match. The operation should
  // keep the same form.
  // CHECK: %{{.*}}, %{{.*}} = "test.native_code_call4"({{%.*}}) : (i8) -> (i32, i32)
  %4, %5 = "test.native_code_call4"(%3) : (i8) -> (i32, i32)
  // CHECK: return %[[A]]
  return %1 : i32
}

// CHECK-LABEL: verifyMultipleNativeCodeCallBinding
func.func @verifyMultipleNativeCodeCallBinding(%arg0 : i32) -> (i32) {
  %0 = "test.op_k"() : () -> (i32)
  %1 = "test.op_k"() : () -> (i32)
  // CHECK: %[[A:.*]] = "test.native_code_call7"(%1) : (i32) -> i32
  // CHECK: %[[A:.*]] = "test.native_code_call7"(%0) : (i32) -> i32
  %2, %3 = "test.native_code_call6"(%0, %1) : (i32, i32) -> (i32, i32)
  return %2 : i32
}

// CHECK-LABEL: verifyAllAttrConstraintOf
func.func @verifyAllAttrConstraintOf() -> (i32, i32, i32) {
  // CHECK: "test.all_attr_constraint_of2"
  %0 = "test.all_attr_constraint_of1"() {attr = [0, 1]} : () -> (i32)
  // CHECK: "test.all_attr_constraint_of1"
  %1 = "test.all_attr_constraint_of1"() {attr = [0, 2]} : () -> (i32)
  // CHECK: "test.all_attr_constraint_of1"
  %2 = "test.all_attr_constraint_of1"() {attr = [-1, 1]} : () -> (i32)
  return %0, %1, %2: i32, i32, i32
}

// CHECK-LABEL: verifyManyArgs
// CHECK-SAME: (%[[ARG:.*]]: i32 loc({{[^)]*}}))
func.func @verifyManyArgs(%arg: i32) {
  // CHECK: "test.many_arguments"(%[[ARG]], %[[ARG]], %[[ARG]], %[[ARG]], %[[ARG]], %[[ARG]], %[[ARG]], %[[ARG]], %[[ARG]])
  // CHECK-SAME: {attr1 = 24 : i64, attr2 = 42 : i64, attr3 = 42 : i64, attr4 = 42 : i64, attr5 = 42 : i64, attr6 = 42 : i64, attr7 = 42 : i64, attr8 = 42 : i64, attr9 = 42 : i64}
  "test.many_arguments"(%arg, %arg, %arg, %arg, %arg, %arg, %arg, %arg, %arg) {
    attr1 = 42, attr2 = 42, attr3 = 42, attr4 = 42, attr5 = 42,
    attr6 = 42, attr7 = 42, attr8 = 42, attr9 = 42
  } : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: verifyEqualArgs
func.func @verifyEqualArgs(%arg0: i32, %arg1: i32) {
  // def TestEqualArgsPattern : Pat<(OpN $a, $a), (OpO $a)>;

  // CHECK: "test.op_o"(%arg0) : (i32) -> i32
  "test.op_n"(%arg0, %arg0) : (i32, i32) -> (i32)

  // CHECK: "test.op_n"(%arg0, %arg1) : (i32, i32) -> i32
  "test.op_n"(%arg0, %arg1) : (i32, i32) -> (i32)

  return
}

// CHECK-LABEL: verifyNestedOpEqualArgs
func.func @verifyNestedOpEqualArgs(
  %arg0: i32, %arg1: i32, %arg2 : i32, %arg3 : i32, %arg4 : i32, %arg5 : i32) {
  // def TestNestedOpEqualArgsPattern :
  //   Pat<(OpN $b, (OpP $a, $b, $c, $d, $e, $f)), (replaceWithValue $b)>;

  // CHECK: %arg1
  %0 = "test.op_p"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
    : (i32, i32, i32, i32, i32, i32) -> (i32)
  %1 = "test.op_n"(%arg1, %0) : (i32, i32) -> (i32)

  // CHECK: test.op_p
  // CHECK: test.op_n
  %2 = "test.op_p"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
    : (i32, i32, i32, i32, i32, i32) -> (i32)
  %3 = "test.op_n"(%arg0, %2) : (i32, i32) -> (i32)

  return
}

// CHECK-LABEL: verifyNestedSameOpAndSameArgEquality
func.func @verifyNestedSameOpAndSameArgEquality(%arg0: i32, %arg1: i32) -> i32 {
  // def TestNestedSameOpAndSameArgEqualityPattern:
  //   Pat<(OpN (OpN $_, $x), $x), (replaceWithValue $x)>;

  %0 = "test.op_n"(%arg1, %arg0) : (i32, i32) -> (i32)
  %1 = "test.op_n"(%0, %arg0) : (i32, i32) -> (i32)
  // CHECK: return %arg0 : i32
  return %1 : i32
}

// CHECK-LABEL: verifyMultipleEqualArgs
func.func @verifyMultipleEqualArgs(
  %arg0: i32, %arg1 : i32, %arg2 : i32, %arg3 : i32, %arg4 : i32) {
  // def TestMultipleEqualArgsPattern :
  //   Pat<(OpP $a, $b, $a, $a, $b, $c), (OpN $c, $b)>;

  // CHECK: "test.op_n"(%arg2, %arg1) : (i32, i32) -> i32
  "test.op_p"(%arg0, %arg1, %arg0, %arg0, %arg1, %arg2) :
    (i32, i32, i32, i32 , i32, i32) -> i32

  // CHECK: test.op_p
  "test.op_p"(%arg0, %arg1, %arg0, %arg0, %arg0, %arg2) :
    (i32, i32, i32, i32 , i32, i32) -> i32

  // CHECK: test.op_p
  "test.op_p"(%arg0, %arg1, %arg1, %arg0, %arg1, %arg2) :
    (i32, i32, i32, i32 , i32, i32) -> i32

   // CHECK: test.op_p
  "test.op_p"(%arg0, %arg1, %arg2, %arg2, %arg3, %arg4) :
    (i32, i32, i32, i32 , i32, i32) -> i32

  return
}

//===----------------------------------------------------------------------===//
// Test Symbol Binding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: symbolBinding
func.func @symbolBinding(%arg0: i32) -> i32 {
  // An op with one use is matched.
  // CHECK: %0 = "test.symbol_binding_b"(%arg0)
  // CHECK: %1 = "test.symbol_binding_c"(%0)
  // CHECK: %2 = "test.symbol_binding_d"(%0, %1) {attr = 42 : i64}
  %0 = "test.symbol_binding_a"(%arg0) {attr = 42} : (i32) -> (i32)

  // An op without any use is not matched.
  // CHECK: "test.symbol_binding_a"(%arg0)
  %1 = "test.symbol_binding_a"(%arg0) {attr = 42} : (i32) -> (i32)

  // CHECK: return %2
  return %0: i32
}

// CHECK-LABEL: symbolBindingNoResult
func.func @symbolBindingNoResult(%arg0: i32) {
  // CHECK: test.symbol_binding_b
  "test.symbol_binding_no_result"(%arg0) : (i32) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test Attributes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: succeedMatchOpAttr
func.func @succeedMatchOpAttr() -> i32 {
  // CHECK: "test.match_op_attribute2"() {default_valued_attr = 3 : i32, more_attr = 4 : i32, optional_attr = 2 : i32, required_attr = 1 : i32}
  %0 = "test.match_op_attribute1"() {required_attr = 1: i32, optional_attr = 2: i32, default_valued_attr = 3: i32, more_attr = 4: i32} : () -> (i32)
  return %0: i32
}

// CHECK-LABEL: succeedMatchMissingOptionalAttr
func.func @succeedMatchMissingOptionalAttr() -> i32 {
  // CHECK: "test.match_op_attribute2"() {default_valued_attr = 3 : i32, more_attr = 4 : i32, required_attr = 1 : i32}
  %0 = "test.match_op_attribute1"() {required_attr = 1: i32, default_valued_attr = 3: i32, more_attr = 4: i32} : () -> (i32)
  return %0: i32
}

// CHECK-LABEL: succeedMatchMissingDefaultValuedAttr
func.func @succeedMatchMissingDefaultValuedAttr() -> i32 {
  // CHECK: "test.match_op_attribute2"() {default_valued_attr = 42 : i32, more_attr = 4 : i32, optional_attr = 2 : i32, required_attr = 1 : i32}
  %0 = "test.match_op_attribute1"() {required_attr = 1: i32, optional_attr = 2: i32, more_attr = 4: i32} : () -> (i32)
  return %0: i32
}

// CHECK-LABEL: failedMatchAdditionalConstraintNotSatisfied
func.func @failedMatchAdditionalConstraintNotSatisfied() -> i32 {
  // CHECK: "test.match_op_attribute1"()
  %0 = "test.match_op_attribute1"() {required_attr = 1: i32, optional_attr = 2: i32, more_attr = 5: i32} : () -> (i32)
  return %0: i32
}

// CHECK-LABEL: verifyConstantAttr
func.func @verifyConstantAttr(%arg0 : i32) -> i32 {
  // CHECK: "test.op_b"(%arg0) {attr = 17 : i32} : (i32) -> i32 loc("a")
  %0 = "test.op_c"(%arg0) : (i32) -> i32 loc("a")
  return %0 : i32
}

// CHECK-LABEL: verifyUnitAttr
func.func @verifyUnitAttr() -> (i32, i32) {
  // Unit attribute present in the matched op is propagated as attr2.
  // CHECK: "test.match_op_attribute4"() {attr1, attr2} : () -> i32
  %0 = "test.match_op_attribute3"() {attr} : () -> i32

  // Since the original op doesn't have the unit attribute, the new op
  // only has the constant-constructed unit attribute attr1.
  // CHECK: "test.match_op_attribute4"() {attr1} : () -> i32
  %1 = "test.match_op_attribute3"() : () -> i32
  return %0, %1 : i32, i32
}

//===----------------------------------------------------------------------===//
// Test Constant Matching
//===----------------------------------------------------------------------===//

// CHECK-LABEL: testConstOp
func.func @testConstOp() -> (i32) {
  // CHECK-NEXT: [[C0:%.+]] = "test.constant"() {value = 1
  %0 = "test.constant"() {value = 1 : i32} : () -> i32

  // CHECK-NEXT: return [[C0]]
  return %0 : i32
}

// CHECK-LABEL: testConstOpUsed
func.func @testConstOpUsed() -> (i32) {
  // CHECK-NEXT: [[C0:%.+]] = "test.constant"() {value = 1
  %0 = "test.constant"() {value = 1 : i32} : () -> i32

  // CHECK-NEXT: [[V0:%.+]] = "test.op_s"([[C0]])
  %1 = "test.op_s"(%0) {value = 1 : i32} : (i32) -> i32

  // CHECK-NEXT: return [[V0]]
  return %1 : i32
}

// CHECK-LABEL: testConstOpReplaced
func.func @testConstOpReplaced() -> (i32) {
  // CHECK-NEXT: [[C0:%.+]] = "test.constant"() {value = 1
  %0 = "test.constant"() {value = 1 : i32} : () -> i32
  %1 = "test.constant"() {value = 2 : i32} : () -> i32

  // CHECK: [[V0:%.+]] = "test.op_s"([[C0]]) {value = 2 : i32}
  %2 = "test.op_r"(%0, %1) : (i32, i32) -> i32

  // CHECK: [[V0]]
  return %2 : i32
}

// CHECK-LABEL: testConstOpMatchFailure
func.func @testConstOpMatchFailure() -> (i64) {
  // CHECK-DAG: [[C0:%.+]] = "test.constant"() {value = 1
  %0 = "test.constant"() {value = 1 : i64} : () -> i64

  // CHECK-DAG: [[C1:%.+]] = "test.constant"() {value = 2
  %1 = "test.constant"() {value = 2 : i64} : () -> i64

  // CHECK: [[V0:%.+]] = "test.op_r"([[C0]], [[C1]])
  %2 = "test.op_r"(%0, %1) : (i64, i64) -> i64

  // CHECK: [[V0]]
  return %2 : i64
}

// CHECK-LABEL: testConstOpMatchNonConst
func.func @testConstOpMatchNonConst(%arg0 : i32) -> (i32) {
  // CHECK-DAG: [[C0:%.+]] = "test.constant"() {value = 1
  %0 = "test.constant"() {value = 1 : i32} : () -> i32

  // CHECK: [[V0:%.+]] = "test.op_r"([[C0]], %arg0)
  %1 = "test.op_r"(%0, %arg0) : (i32, i32) -> i32

  // CHECK: [[V0]]
  return %1 : i32
}



//===----------------------------------------------------------------------===//
// Test Enum Attributes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: verifyI32EnumAttr
func.func @verifyI32EnumAttr() -> i32 {
  // CHECK: "test.i32_enum_attr"() {attr = 10 : i32}
  %0 = "test.i32_enum_attr"() {attr = 5: i32} : () -> i32
  return %0 : i32
}

// CHECK-LABEL: verifyI64EnumAttr
func.func @verifyI64EnumAttr() -> i32 {
  // CHECK: "test.i64_enum_attr"() {attr = 10 : i64}
  %0 = "test.i64_enum_attr"() {attr = 5: i64} : () -> i32
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// Test ElementsAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: rewrite_i32elementsattr
func.func @rewrite_i32elementsattr() -> () {
  // CHECK: attr = dense<0> : tensor<i32>
  "test.i32ElementsAttr"() {attr = dense<[3, 5]>:tensor<2xi32>} : () -> ()
  return
}

// CHECK-LABEL: rewrite_f64elementsattr
func.func @rewrite_f64elementsattr() -> () {
  "test.float_elements_attr"() {
    // Should match
    // CHECK: scalar_f32_attr = dense<[5.000000e+00, 6.000000e+00]> : tensor<2xf32>
    scalar_f32_attr = dense<[3.0, 4.0]> : tensor<2xf32>,
    tensor_f64_attr = dense<6.0> : tensor<4x8xf64>
  } : () -> ()

  "test.float_elements_attr"() {
    // Should not match
    // CHECK: scalar_f32_attr = dense<7.000000e+00> : tensor<2xf32>
    scalar_f32_attr = dense<7.0> : tensor<2xf32>,
    tensor_f64_attr = dense<3.0> : tensor<4x8xf64>
  } : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test Multi-result Ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @useMultiResultOpToReplaceWhole
func.func @useMultiResultOpToReplaceWhole() -> (i32, f32, f32) {
  // CHECK: %[[A:.*]], %[[B:.*]], %[[C:.*]] = "test.another_three_result"()
  // CHECK: return %[[A]], %[[B]], %[[C]]
  %0:3 = "test.three_result"() {kind = 1} : () -> (i32, f32, f32)
  return %0#0, %0#1, %0#2 : i32, f32, f32
}

// CHECK-LABEL: @useMultiResultOpToReplacePartial1
func.func @useMultiResultOpToReplacePartial1() -> (i32, f32, f32) {
  // CHECK: %[[A:.*]], %[[B:.*]] = "test.two_result"()
  // CHECK: %[[C:.*]] = "test.one_result1"()
  // CHECK: return %[[A]], %[[B]], %[[C]]
  %0:3 = "test.three_result"() {kind = 2} : () -> (i32, f32, f32)
  return %0#0, %0#1, %0#2 : i32, f32, f32
}

// CHECK-LABEL: @useMultiResultOpToReplacePartial2
func.func @useMultiResultOpToReplacePartial2() -> (i32, f32, f32) {
  // CHECK: %[[A:.*]] = "test.one_result2"()
  // CHECK: %[[B:.*]], %[[C:.*]] = "test.another_two_result"()
  // CHECK: return %[[A]], %[[B]], %[[C]]
  %0:3 = "test.three_result"() {kind = 3} : () -> (i32, f32, f32)
  return %0#0, %0#1, %0#2 : i32, f32, f32
}

// CHECK-LABEL: @useMultiResultOpResultsSeparately
func.func @useMultiResultOpResultsSeparately() -> (i32, f32, f32) {
  // CHECK: %[[A:.*]], %[[B:.*]] = "test.two_result"()
  // CHECK: %[[C:.*]] = "test.one_result1"()
  // CHECK: %[[D:.*]], %[[E:.*]] = "test.two_result"()
  // CHECK: return %[[A]], %[[C]], %[[E]]
  %0:3 = "test.three_result"() {kind = 4} : () -> (i32, f32, f32)
  return %0#0, %0#1, %0#2 : i32, f32, f32
}

// CHECK-LABEL: @constraintOnSourceOpResult
func.func @constraintOnSourceOpResult() -> (i32, f32, i32) {
  // CHECK: %[[A:.*]], %[[B:.*]] = "test.two_result"()
  // CHECK: %[[C:.*]] = "test.one_result2"()
  // CHECK: %[[D:.*]] = "test.one_result1"()
  // CHECK: return %[[A]], %[[B]], %[[C]]
  %0:2 = "test.two_result"() {kind = 5} : () -> (i32, f32)
  %1:2 = "test.two_result"() {kind = 5} : () -> (i32, f32)
  return %0#0, %0#1, %1#0 : i32, f32, i32
}

// CHECK-LABEL: @useAuxiliaryOpToReplaceMultiResultOp
func.func @useAuxiliaryOpToReplaceMultiResultOp() -> (i32, f32, f32) {
  // An auxiliary op is generated to help building the op for replacing the
  // matched op.
  // CHECK: %[[A:.*]], %[[B:.*]] = "test.two_result"()

  // CHECK: %[[C:.*]] = "test.one_result3"(%[[B]])
  // CHECK: %[[D:.*]], %[[E:.*]] = "test.another_two_result"()
  // CHECK: return %[[C]], %[[D]], %[[E]]
  %0:3 = "test.three_result"() {kind = 6} : () -> (i32, f32, f32)
  return %0#0, %0#1, %0#2 : i32, f32, f32
}

//===----------------------------------------------------------------------===//
// Test Multi-result Ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @replaceOneVariadicOutOneVariadicInOp
func.func @replaceOneVariadicOutOneVariadicInOp(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32, i32, i32, i32, i32) {
  // CHECK: %[[cnt1:.*]] = "test.one_variadic_out_one_variadic_in2"(%arg0)
  // CHECK: %[[cnt2:.*]]:2 = "test.one_variadic_out_one_variadic_in2"(%arg0, %arg1)
  // CHECK: %[[cnt3:.*]]:3 = "test.one_variadic_out_one_variadic_in2"(%arg0, %arg1, %arg2)
  // CHECK: return %[[cnt1]], %[[cnt2]]#0, %[[cnt2]]#1, %[[cnt3]]#0, %[[cnt3]]#1, %[[cnt3]]#2

  %0   = "test.one_variadic_out_one_variadic_in1"(%arg0) : (i32) -> (i32)
  %1:2 = "test.one_variadic_out_one_variadic_in1"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
  %2:3 = "test.one_variadic_out_one_variadic_in1"(%arg0, %arg1, %arg2) : (i32, i32, i32) -> (i32, i32, i32)
  return %0, %1#0, %1#1, %2#0, %2#1, %2#2 : i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: @replaceMixedVariadicInputOp
func.func @replaceMixedVariadicInputOp(%arg0: i32, %arg1: f32, %arg2: i32) -> () {
  // CHECK: "test.mixed_variadic_in2"(%arg1)
  // CHECK: "test.mixed_variadic_in2"(%arg0, %arg1, %arg2)
  // CHECK: "test.mixed_variadic_in2"(%arg0, %arg0, %arg1, %arg2, %arg2)

  "test.mixed_variadic_in1"(%arg1) : (f32) -> ()
  "test.mixed_variadic_in1"(%arg0, %arg1, %arg2) : (i32, f32, i32) -> ()
  "test.mixed_variadic_in1"(%arg0, %arg0, %arg1, %arg2, %arg2) : (i32, i32, f32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @replaceMixedVariadicOutputOp
func.func @replaceMixedVariadicOutputOp() -> (f32, i32, f32, i32, i32, i32, f32, i32, i32) {
  // CHECK: %[[cnt1:.*]] = "test.mixed_variadic_out2"()
  // CHECK: %[[cnt3_a:.*]], %[[cnt3_b:.*]], %[[cnt3_c:.*]] = "test.mixed_variadic_out2"()
  // CHECK: %[[cnt5_a:.*]]:2, %[[cnt5_b:.*]], %[[cnt5_c:.*]]:2 = "test.mixed_variadic_out2"()
  // CHECK: return %[[cnt1]], %[[cnt3_a]], %[[cnt3_b]], %[[cnt3_c]], %[[cnt5_a]]#0, %[[cnt5_a]]#1, %[[cnt5_b]], %[[cnt5_c]]#0, %[[cnt5_c]]#1

  %0   = "test.mixed_variadic_out1"() : () -> (f32)
  %1:3 = "test.mixed_variadic_out1"() : () -> (i32, f32, i32)
  %2:5 = "test.mixed_variadic_out1"() : () -> (i32, i32, f32, i32, i32)
  return %0, %1#0, %1#1, %1#2, %2#0, %2#1, %2#2, %2#3, %2#4 : f32, i32, f32, i32, i32, i32, f32, i32, i32
}

// CHECK-LABEL: @generateVariadicOutputOpInNestedPattern
func.func @generateVariadicOutputOpInNestedPattern() -> (i32) {
  // CHECK: %[[cnt5_a:.*]], %[[cnt5_b:.*]]:2, %[[cnt5_c:.*]]:2 = "test.mixed_variadic_out3"()
  // CHECK: %[[res:.*]] = "test.mixed_variadic_in3"(%[[cnt5_a]], %[[cnt5_b]]#0, %[[cnt5_b]]#1, %[[cnt5_c]]#0, %[[cnt5_c]]#1)
  // CHECK: return %[[res]]

  %0 = "test.one_i32_out"() : () -> (i32)
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// Test that natives calls are only called once during rewrites.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: redundantTest
func.func @redundantTest(%arg0: i32) -> i32 {
  %0 = "test.op_m"(%arg0) : (i32) -> i32
  // CHECK: "test.op_m"(%arg0) {optional_attr = 314159265 : i32} : (i32) -> i32
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// Test either directive
//===----------------------------------------------------------------------===//

// CHECK: @either_dag_leaf_only
func.func @either_dag_leaf_only_1(%arg0 : i32, %arg1 : i16, %arg2 : i8) -> () {
  // CHECK: "test.either_op_b"(%arg1) : (i16) -> i32
  %0 = "test.either_op_a"(%arg0, %arg1, %arg2) : (i32, i16, i8) -> i32
  // CHECK: "test.either_op_b"(%arg1) : (i16) -> i32
  %1 = "test.either_op_a"(%arg1, %arg0, %arg2) : (i16, i32, i8) -> i32
  return
}

// CHECK: @either_dag_leaf_dag_node
func.func @either_dag_leaf_dag_node(%arg0 : i32, %arg1 : i16, %arg2 : i8) -> () {
  %0 = "test.either_op_b"(%arg0) : (i32) -> i32
  // CHECK: "test.either_op_b"(%arg1) : (i16) -> i32
  %1 = "test.either_op_a"(%0, %arg1, %arg2) : (i32, i16, i8) -> i32
  // CHECK: "test.either_op_b"(%arg1) : (i16) -> i32
  %2 = "test.either_op_a"(%arg1, %0, %arg2) : (i16, i32, i8) -> i32
  return
}

// CHECK: @either_dag_node_dag_node
func.func @either_dag_node_dag_node(%arg0 : i32, %arg1 : i16, %arg2 : i8) -> () {
  %0 = "test.either_op_b"(%arg0) : (i32) -> i32
  %1 = "test.either_op_b"(%arg1) : (i16) -> i32
  // CHECK: "test.either_op_b"(%arg1) : (i16) -> i32
  %2 = "test.either_op_a"(%0, %1, %arg2) : (i32, i32, i8) -> i32
  // CHECK: "test.either_op_b"(%arg1) : (i16) -> i32
  %3 = "test.either_op_a"(%1, %0, %arg2) : (i32, i32, i8) -> i32
  return
}

//===----------------------------------------------------------------------===//
// Test that ops without type deduction can be created with type builders.
//===----------------------------------------------------------------------===//

func.func @explicitReturnTypeTest(%arg0 : i64) -> i8 {
  %0 = "test.source_op"(%arg0) {tag = 11 : i32} : (i64) -> i8
  // CHECK: "test.op_x"(%arg0) : (i64) -> i32
  // CHECK: "test.op_x"(%0) : (i32) -> i8
  return %0 : i8
}

func.func @returnTypeBuilderTest(%arg0 : i1) -> i8 {
  %0 = "test.source_op"(%arg0) {tag = 22 : i32} : (i1) -> i8
  // CHECK: "test.op_x"(%arg0) : (i1) -> i1
  // CHECK: "test.op_x"(%0) : (i1) -> i8
  return %0 : i8
}

func.func @multipleReturnTypeBuildTest(%arg0 : i1) -> i1 {
  %0 = "test.source_op"(%arg0) {tag = 33 : i32} : (i1) -> i1
  // CHECK: "test.one_to_two"(%arg0) : (i1) -> (i64, i32)
  // CHECK: "test.op_x"(%0#0) : (i64) -> i32
  // CHECK: "test.op_x"(%0#1) : (i32) -> i64
  // CHECK: "test.two_to_one"(%1, %2) : (i32, i64) -> i1
  return %0 : i1
}

func.func @copyValueType(%arg0 : i8) -> i32 {
  %0 = "test.source_op"(%arg0) {tag = 44 : i32} : (i8) -> i32
  // CHECK: "test.op_x"(%arg0) : (i8) -> i8
  // CHECK: "test.op_x"(%0) : (i8) -> i32
  return %0 : i32
}

func.func @multipleReturnTypeDifferent(%arg0 : i1) -> i64 {
  %0 = "test.source_op"(%arg0) {tag = 55 : i32} : (i1) -> i64
  // CHECK: "test.one_to_two"(%arg0) : (i1) -> (i1, i64)
  // CHECK: "test.two_to_one"(%0#0, %0#1) : (i1, i64) -> i64
  return %0 : i64
}

//===----------------------------------------------------------------------===//
// Test that multiple trailing directives can be mixed in patterns.
//===----------------------------------------------------------------------===//

func.func @returnTypeAndLocation(%arg0 : i32) -> i1 {
  %0 = "test.source_op"(%arg0) {tag = 66 : i32} : (i32) -> i1
  // CHECK: "test.op_x"(%arg0) : (i32) -> i32 loc("loc1")
  // CHECK: "test.op_x"(%arg0) : (i32) -> i32 loc("loc2")
  // CHECK: "test.two_to_one"(%0, %1) : (i32, i32) -> i1
  return %0 : i1
}

//===----------------------------------------------------------------------===//
// Test that patterns can create ConstantStrAttr
//===----------------------------------------------------------------------===//

func.func @testConstantStrAttr() -> () {
  // CHECK: test.has_str_value {value = "foo"}
  test.no_str_value {value = "bar"}
  return
}
