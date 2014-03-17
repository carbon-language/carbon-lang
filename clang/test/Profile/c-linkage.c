// Check the data structures emitted by instrumentation.
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-linkage.c %s -o - -emit-llvm -fprofile-instr-generate | FileCheck %s

// CHECK: @__llvm_pgo_counters_foo = global [1 x i64] zeroinitializer, section "__DATA,__llvm_pgo_cnts", align 8
// CHECK: @__llvm_pgo_name_foo = constant [3 x i8] c"foo", section "__DATA,__llvm_pgo_names", align 1
// CHECK: @__llvm_pgo_data_foo = constant { i32, i32, i8*, i64* } { i32 3, i32 1, i8* getelementptr inbounds ([3 x i8]* @__llvm_pgo_name_foo, i32 0, i32 0), i64* getelementptr inbounds ([1 x i64]* @__llvm_pgo_counters_foo, i32 0, i32 0) }, section "__DATA,__llvm_pgo_data", align 8
void foo(void) { }

// CHECK: @__llvm_pgo_counters_foo_weak = weak global [5 x i64] zeroinitializer, section "__DATA,__llvm_pgo_cnts", align 8
// CHECK: @__llvm_pgo_name_foo_weak = weak constant [8 x i8] c"foo_weak", section "__DATA,__llvm_pgo_names", align 1
// CHECK: @__llvm_pgo_data_foo_weak = weak constant { i32, i32, i8*, i64* } { i32 8, i32 5, i8* getelementptr inbounds ([8 x i8]* @__llvm_pgo_name_foo_weak, i32 0, i32 0), i64* getelementptr inbounds ([5 x i64]* @__llvm_pgo_counters_foo_weak, i32 0, i32 0) }, section "__DATA,__llvm_pgo_data", align 8
void foo_weak(void) __attribute__((weak));
void foo_weak(void) { if (0){} if (0){} if (0){} if (0){} }

// CHECK: @__llvm_pgo_counters_main = global [1 x i64] zeroinitializer, section "__DATA,__llvm_pgo_cnts", align 8
// CHECK: @__llvm_pgo_name_main = constant [4 x i8] c"main", section "__DATA,__llvm_pgo_names", align 1
// CHECK: @__llvm_pgo_data_main = constant { i32, i32, i8*, i64* } { i32 4, i32 1, i8* getelementptr inbounds ([4 x i8]* @__llvm_pgo_name_main, i32 0, i32 0), i64* getelementptr inbounds ([1 x i64]* @__llvm_pgo_counters_main, i32 0, i32 0) }, section "__DATA,__llvm_pgo_data", align 8
static void foo_internal(void);
int main(void) {
  foo();
  foo_internal();
  foo_weak();
  return 0;
}

// CHECK: @__llvm_pgo_counters_foo_internal = internal global [3 x i64] zeroinitializer, section "__DATA,__llvm_pgo_cnts", align 8
// CHECK: @__llvm_pgo_name_foo_internal = internal constant [24 x i8] c"c-linkage.c:foo_internal", section "__DATA,__llvm_pgo_names", align 1
// CHECK: @__llvm_pgo_data_foo_internal = internal constant { i32, i32, i8*, i64* } { i32 24, i32 3, i8* getelementptr inbounds ([24 x i8]* @__llvm_pgo_name_foo_internal, i32 0, i32 0), i64* getelementptr inbounds ([3 x i64]* @__llvm_pgo_counters_foo_internal, i32 0, i32 0) }, section "__DATA,__llvm_pgo_data", align 8
static void foo_internal(void) { if (0){} if (0){} }

// CHECK: @llvm.used = appending global [4 x i8*] [i8* bitcast ({ i32, i32, i8*, i64* }* @__llvm_pgo_data_foo to i8*), i8* bitcast ({ i32, i32, i8*, i64* }* @__llvm_pgo_data_foo_weak to i8*), i8* bitcast ({ i32, i32, i8*, i64* }* @__llvm_pgo_data_main to i8*), i8* bitcast ({ i32, i32, i8*, i64* }* @__llvm_pgo_data_foo_internal to i8*)], section "llvm.metadata"
