; RUN: opt %s -mtriple=aarch64-none-linux-gnu -global-merge -S -o - | FileCheck %s
; RUN: opt %s -mtriple=aarch64-none-linux-gnu -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s

; RUN: opt %s -mtriple=arm64-linux-gnuabi -global-merge -S -o - | FileCheck %s
; RUN: opt %s -mtriple=arm64-linux-gnuabi -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s

; RUN: opt %s -mtriple=arm64-apple-ios -global-merge -S -o - | FileCheck %s
; RUN: opt %s -mtriple=arm64-apple-ios -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s

@m = internal global i32 0, align 4
@n = internal global i32 0, align 4

; CHECK: @_MergedGlobals = internal global { i32, i32 } zeroinitializer

define void @f1(i32 %a1, i32 %a2) {
; CHECK-LABEL: @f1
; CHECK: getelementptr inbounds ({ i32, i32 }* @_MergedGlobals, i32 0, i32 0)
; CHECK: getelementptr inbounds ({ i32, i32 }* @_MergedGlobals, i32 0, i32 1)
  store i32 %a1, i32* @m, align 4
  store i32 %a2, i32* @n, align 4
  ret void
}
