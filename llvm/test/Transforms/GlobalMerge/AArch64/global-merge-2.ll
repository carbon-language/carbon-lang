; RUN: opt %s -mtriple=aarch64-none-linux-gnu -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s
; RUN: opt %s -mtriple=arm64-linux-gnuabi -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s
; RUN: opt %s -mtriple=arm64-apple-ios -global-merge -global-merge-on-external -global-merge-aligned -S -o - | FileCheck %s

@x = global i32 0, align 4
@y = global i32 0, align 4
@z = global i32 0, align 4

; CHECK: @_MergedGlobals_x = global { i32, i32, i32 } zeroinitializer, align 16
; CHECK: @x = alias getelementptr inbounds ({ i32, i32, i32 }* @_MergedGlobals_x, i32 0, i32 0)
; CHECK: @y = alias getelementptr inbounds ({ i32, i32, i32 }* @_MergedGlobals_x, i32 0, i32 1)
; CHECK: @z = alias getelementptr inbounds ({ i32, i32, i32 }* @_MergedGlobals_x, i32 0, i32 2)

define void @f1(i32 %a1, i32 %a2) {
; CHECK-LABEL: @f1
; CHECK: getelementptr inbounds ({ i32, i32, i32 }* @_MergedGlobals_x, i32 0, i32 0)
; CHECK: getelementptr inbounds ({ i32, i32, i32 }* @_MergedGlobals_x, i32 0, i32 1)
  store i32 %a1, i32* @x, align 4
  store i32 %a2, i32* @y, align 4
  ret void
}

define void @g1(i32 %a1, i32 %a2) {
; CHECK-LABEL: @g1
; CHECK: getelementptr inbounds ({ i32, i32, i32 }* @_MergedGlobals_x, i32 0, i32 1)
; CHECK: getelementptr inbounds ({ i32, i32, i32 }* @_MergedGlobals_x, i32 0, i32 2)
  store i32 %a1, i32* @y, align 4
  store i32 %a2, i32* @z, align 4
  ret void
}
