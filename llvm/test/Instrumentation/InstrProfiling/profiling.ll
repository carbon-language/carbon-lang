; RUN: opt < %s -instrprof -S | FileCheck %s

target triple = "x86_64-apple-macosx10.10.0"

@__prf_nm_foo = hidden constant [3 x i8] c"foo"
; CHECK: @__prf_nm_foo = hidden constant [3 x i8] c"foo", section "__DATA,__llvm_prf_names", align 1
@__prf_nm_bar = hidden constant [4 x i8] c"bar\00"
; CHECK: @__prf_nm_bar = hidden constant [4 x i8] c"bar\00", section "__DATA,__llvm_prf_names", align 1
@__prf_nm_baz = hidden constant [3 x i8] c"baz"
; CHECK: @__prf_nm_baz = hidden constant [3 x i8] c"baz", section "__DATA,__llvm_prf_names", align 1

; CHECK: @__prf_cn_foo = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__prf_dt_foo = hidden {{.*}}, section "__DATA,__llvm_prf_data", align 8
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__prf_nm_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @__prf_cn_bar = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__prf_dt_bar = hidden {{.*}}, section "__DATA,__llvm_prf_data", align 8
define void @bar() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__prf_nm_bar, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @__prf_cn_baz = hidden global [3 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__prf_dt_baz = hidden {{.*}}, section "__DATA,__llvm_prf_data", align 8
define void @baz() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__prf_nm_baz, i32 0, i32 0), i64 0, i32 3, i32 0)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__prf_nm_baz, i32 0, i32 0), i64 0, i32 3, i32 1)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__prf_nm_baz, i32 0, i32 0), i64 0, i32 3, i32 2)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; CHECK: @__llvm_profile_runtime = external global i32
; CHECK: @llvm.used = appending global {{.*}} @__prf_dt_foo {{.*}} @__prf_dt_bar {{.*}} @__prf_dt_baz {{.*}} section "llvm.metadata"
