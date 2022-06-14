// RUN: mlir-opt %s | mlir-opt | FileCheck %s

emitc.include <"test.h">
emitc.include "test.h"

// CHECK-LABEL: func @f(%{{.*}}: i32, %{{.*}}: !emitc.opaque<"int32_t">) {
func.func @f(%arg0: i32, %f: !emitc.opaque<"int32_t">) {
  %1 = "emitc.call"() {callee = "blah"} : () -> i64
  emitc.call "foo" (%1) {args = [
    0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
  ]} : (i64) -> ()
  return
}

func.func @cast(%arg0: i32) {
  %1 = emitc.cast %arg0: i32 to f32
  return
}

func.func @c() {
  %1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  return
}

func.func @a(%arg0: i32, %arg1: i32) {
  %1 = "emitc.apply"(%arg0) {applicableOperator = "&"} : (i32) -> !emitc.ptr<i32>
  %2 = emitc.apply "&"(%arg1) : (i32) -> !emitc.ptr<i32>
  return
}
