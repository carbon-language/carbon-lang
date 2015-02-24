;; Check that runtime symbols get appropriate linkage.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -S | FileCheck %s

@__llvm_profile_name_foo = hidden constant [3 x i8] c"foo"
@__llvm_profile_name_foo_weak = weak hidden constant [8 x i8] c"foo_weak"
@"__llvm_profile_name_linkage.ll:foo_internal" = internal constant [23 x i8] c"linkage.ll:foo_internal"
@__llvm_profile_name_foo_inline = linkonce_odr hidden constant [10 x i8] c"foo_inline"

; CHECK: @__llvm_profile_counters_foo = hidden global
; CHECK: @__llvm_profile_data_foo = hidden constant
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8]* @__llvm_profile_name_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @__llvm_profile_counters_foo_weak = weak hidden global
; CHECK: @__llvm_profile_data_foo_weak = weak hidden constant
define weak void @foo_weak() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([8 x i8]* @__llvm_profile_name_foo_weak, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @"__llvm_profile_counters_linkage.ll:foo_internal" = internal global
; CHECK: @"__llvm_profile_data_linkage.ll:foo_internal" = internal constant
define internal void @foo_internal() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([23 x i8]* @"__llvm_profile_name_linkage.ll:foo_internal", i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @__llvm_profile_counters_foo_inline = linkonce_odr hidden global
; CHECK: @__llvm_profile_data_foo_inline = linkonce_odr hidden constant
define linkonce_odr void @foo_inline() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([10 x i8]* @__llvm_profile_name_foo_inline, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; CHECK: @__llvm_profile_runtime = external global i32

; CHECK: define linkonce_odr i32 @__llvm_profile_runtime_user() {{.*}} {
; CHECK:   %[[REG:.*]] = load i32* @__llvm_profile_runtime
; CHECK:   ret i32 %[[REG]]
; CHECK: }
