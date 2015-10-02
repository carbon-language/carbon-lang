// RUN: %clang_cc1 -triple wasm32-unknown-unknown -O3 -emit-llvm -o - %s \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -O3 -emit-llvm -o - %s \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY64

__SIZE_TYPE__ f0(void) {
  return __builtin_wasm_page_size();
// WEBASSEMBLY32: call {{i.*}} @llvm.wasm.page.size.i32()
// WEBASSEMBLY64: call {{i.*}} @llvm.wasm.page.size.i64()
}

__SIZE_TYPE__ f1(void) {
  return __builtin_wasm_memory_size();
// WEBASSEMBLY32: call {{i.*}} @llvm.wasm.memory.size.i32()
// WEBASSEMBLY64: call {{i.*}} @llvm.wasm.memory.size.i64()
}
