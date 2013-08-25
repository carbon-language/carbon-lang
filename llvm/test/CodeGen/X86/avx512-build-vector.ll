; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

; CHECK-LABEL: test1
; CHECK: vpxord
; CHECK: ret
define <16 x i32> @test1(i32* %x) {
   %y = load i32* %x, align 4
   %res = insertelement <16 x i32>zeroinitializer, i32 %y, i32 4
   ret <16 x i32>%res
}

; CHECK-LABEL: test2
; CHECK: vpaddd LCP{{.*}}(%rip){1to16}
; CHECK: ret
define <16 x i32> @test2(<16 x i32> %x) {
   %res = add <16 x i32><i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, %x
   ret <16 x i32>%res
}