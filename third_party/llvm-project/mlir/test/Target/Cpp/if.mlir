// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func @test_if(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
     %0 = emitc.call "func_const"(%arg1) : (f32) -> i32
  }
  return
}
// CPP-DEFAULT: void test_if(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: if ([[V0]]) {
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = func_const([[V1]]);
// CPP-DEFAULT-NEXT: ;
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_if(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: if ([[V0]]) {
// CPP-DECLTOP-NEXT: [[V2]] = func_const([[V1]]);
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: return;


func @test_if_else(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
    %0 = emitc.call "func_true"(%arg1) : (f32) -> i32
  } else {
    %0 = emitc.call "func_false"(%arg1) : (f32) -> i32
  }
  return
}
// CPP-DEFAULT: void test_if_else(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: if ([[V0]]) {
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = func_true([[V1]]);
// CPP-DEFAULT-NEXT: ;
// CPP-DEFAULT-NEXT: } else {
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = func_false([[V1]]);
// CPP-DEFAULT-NEXT: ;
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_if_else(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: if ([[V0]]) {
// CPP-DECLTOP-NEXT: [[V2]] = func_true([[V1]]);
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: } else {
// CPP-DECLTOP-NEXT: [[V3]] = func_false([[V1]]);
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: return;


func @test_if_yield(%arg0: i1, %arg1: f32) {
  %0 = arith.constant 0 : i8
  %x, %y = scf.if %arg0 -> (i32, f64) {
    %1 = emitc.call "func_true_1"(%arg1) : (f32) -> i32
    %2 = emitc.call "func_true_2"(%arg1) : (f32) -> f64
    scf.yield %1, %2 : i32, f64
  } else {
    %1 = emitc.call "func_false_1"(%arg1) : (f32) -> i32
    %2 = emitc.call "func_false_2"(%arg1) : (f32) -> f64
    scf.yield %1, %2 : i32, f64
  }
  return
}
// CPP-DEFAULT: void test_if_yield(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: int8_t [[V2:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: double [[V4:[^ ]*]];
// CPP-DEFAULT-NEXT: if ([[V0]]) {
// CPP-DEFAULT-NEXT: int32_t [[V5:[^ ]*]] = func_true_1([[V1]]);
// CPP-DEFAULT-NEXT: double [[V6:[^ ]*]] = func_true_2([[V1]]);
// CPP-DEFAULT-NEXT: [[V3]] = [[V5]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V6]];
// CPP-DEFAULT-NEXT: } else {
// CPP-DEFAULT-NEXT: int32_t [[V7:[^ ]*]] = func_false_1([[V1]]);
// CPP-DEFAULT-NEXT: double [[V8:[^ ]*]] = func_false_2([[V1]]);
// CPP-DEFAULT-NEXT: [[V3]] = [[V7]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V8]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_if_yield(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int8_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: double [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V5:[^ ]*]];
// CPP-DECLTOP-NEXT: double [[V6:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V7:[^ ]*]];
// CPP-DECLTOP-NEXT: double [[V8:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V2]] = 0;
// CPP-DECLTOP-NEXT: if ([[V0]]) {
// CPP-DECLTOP-NEXT: [[V5]] = func_true_1([[V1]]);
// CPP-DECLTOP-NEXT: [[V6]] = func_true_2([[V1]]);
// CPP-DECLTOP-NEXT: [[V3]] = [[V5]];
// CPP-DECLTOP-NEXT: [[V4]] = [[V6]];
// CPP-DECLTOP-NEXT: } else {
// CPP-DECLTOP-NEXT: [[V7]] = func_false_1([[V1]]);
// CPP-DECLTOP-NEXT: [[V8]] = func_false_2([[V1]]);
// CPP-DECLTOP-NEXT: [[V3]] = [[V7]];
// CPP-DECLTOP-NEXT: [[V4]] = [[V8]];
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: return;
