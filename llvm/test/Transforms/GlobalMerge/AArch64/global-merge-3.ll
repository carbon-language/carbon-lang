; RUN: opt %s -mtriple=aarch64-none-linux-gnu -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s
; RUN: opt %s -mtriple=arm64-linux-gnuabi -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s
; RUN: opt %s -mtriple=arm64-apple-ios -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s

@x = global [1000 x i32] zeroinitializer, align 1
@y = global [1000 x i32] zeroinitializer, align 1
@z = internal global i32 1, align 4

; CHECK: @_MergedGlobals_x = global { i32, [1000 x i32] } { i32 1, [1000 x i32] zeroinitializer }, align 4096
; CHECK: @_MergedGlobals_y = global { [1000 x i32] } zeroinitializer, align 4096

; CHECK: @x = alias getelementptr inbounds ({ i32, [1000 x i32] }* @_MergedGlobals_x, i32 0, i32 1)
; CHECK: @y = alias getelementptr inbounds ({ [1000 x i32] }* @_MergedGlobals_y, i32 0, i32 0)

define void @f1(i32 %a1, i32 %a2, i32 %a3) {
; CHECK-LABEL: @f1
; CHECK: %x3 = getelementptr inbounds [1000 x i32]* getelementptr inbounds ({ i32, [1000 x i32] }* @_MergedGlobals_x, i32 0, i32 1), i32 0, i64 3
; CHECK: %y3 = getelementptr inbounds [1000 x i32]* getelementptr inbounds ({ [1000 x i32] }* @_MergedGlobals_y, i32 0, i32 0), i32 0, i64 3
; CHECK: store i32 %a3, i32* getelementptr inbounds ({ i32, [1000 x i32] }* @_MergedGlobals_x, i32 0, i32 0), align 4

  %x3 = getelementptr inbounds [1000 x i32]* @x, i32 0, i64 3
  %y3 = getelementptr inbounds [1000 x i32]* @y, i32 0, i64 3
  store i32 %a1, i32* %x3, align 4
  store i32 %a2, i32* %y3, align 4
  store i32 %a3, i32* @z, align 4
  ret void
}
