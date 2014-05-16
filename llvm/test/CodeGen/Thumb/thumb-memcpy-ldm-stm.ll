; RUN: llc -mtriple=thumbv6m-eabi %s -o - | FileCheck %s

@d = external global [64 x i32]
@s = external global [64 x i32]

; Function Attrs: nounwind
define void @t1() #0 {
entry:
; CHECK: ldr [[REG0:r[0-9]]],
; CHECK: ldm [[REG0]]!,
; CHECK: ldr [[REG1:r[0-9]]],
; CHECK: stm [[REG1]]!,
; CHECK: subs [[REG0]], #32
; CHECK-NEXT: ldrb
; CHECK: subs [[REG1]], #32
; CHECK-NEXT: strb
    tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast ([64 x i32]* @s to i8*), i8* bitcast ([64 x i32]* @d to i8*), i32 33, i32 4, i1 false)
    ret void
}

; Function Attrs: nounwind
define void @t2() #0 {
entry:
; CHECK: ldr [[REG0:r[0-9]]],
; CHECK: ldm [[REG0]]!,
; CHECK: ldr [[REG1:r[0-9]]],
; CHECK: stm [[REG1]]!,
; CHECK: ldrh
; CHECK: ldrb
; CHECK: strb
; CHECK: strh
    tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast ([64 x i32]* @s to i8*), i8* bitcast ([64 x i32]* @d to i8*), i32 15, i32 4, i1 false)
    ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1) #1
