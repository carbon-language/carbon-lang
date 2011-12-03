; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-darwin | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=THUMB

; Very basic fast-isel functionality.
define i32 @add(i32 %a, i32 %b) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %tmp = load i32* %a.addr
  %tmp1 = load i32* %b.addr
  %add = add nsw i32 %tmp, %tmp1
  ret i32 %add
}

; Check truncate to bool
define void @test1(i32 %tmp) nounwind {
entry:
%tobool = trunc i32 %tmp to i1
br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
call void @test1(i32 0)
br label %if.end

if.end:                                           ; preds = %if.then, %entry
ret void
; ARM: test1:
; ARM: tst r0, #1
; THUMB: test1:
; THUMB: tst.w r0, #1
}

; Check some simple operations with immediates
define void @test2(i32 %tmp, i32* %ptr) nounwind {
; THUMB: test2:
; ARM: test2:

b1:
  %a = add i32 %tmp, 4096
  store i32 %a, i32* %ptr
  br label %b2

; THUMB: add.w {{.*}} #4096
; ARM: add {{.*}} #4096

b2:
  %b = add i32 %tmp, 4095
  store i32 %b, i32* %ptr
  br label %b3
; THUMB: addw {{.*}} #4095
; ARM: movw {{.*}} #4095
; ARM: add

b3:
  %c = or i32 %tmp, 4
  store i32 %c, i32* %ptr
  ret void

; THUMB: orr {{.*}} #4
; ARM: orr {{.*}} #4
}

define void @test3(i32 %tmp, i32* %ptr1, i16* %ptr2, i8* %ptr3) nounwind {
; THUMB: test3:
; ARM: test3:

bb1:
  %a1 = trunc i32 %tmp to i16
  %a2 = trunc i16 %a1 to i8
  %a3 = trunc i8 %a2 to i1
  %a4 = zext i1 %a3 to i8
  store i8 %a4, i8* %ptr3
  %a5 = zext i8 %a4 to i16
  store i16 %a5, i16* %ptr2
  %a6 = zext i16 %a5 to i32
  store i32 %a6, i32* %ptr1
  br label %bb2

; THUMB: and
; THUMB: strb
; THUMB: uxtb
; THUMB: strh
; THUMB: uxth
; ARM: and
; ARM: strb
; ARM: uxtb
; ARM: strh
; ARM: uxth

bb2:
  %b1 = trunc i32 %tmp to i16
  %b2 = trunc i16 %b1 to i8
  store i8 %b2, i8* %ptr3
  %b3 = sext i8 %b2 to i16
  store i16 %b3, i16* %ptr2
  %b4 = sext i16 %b3 to i32
  store i32 %b4, i32* %ptr1
  br label %bb3

; THUMB: strb
; THUMB: sxtb
; THUMB: strh
; THUMB: sxth
; ARM: strb
; ARM: sxtb
; ARM: strh
; ARM: sxth

bb3:
  %c1 = load i8* %ptr3
  %c2 = load i16* %ptr2
  %c3 = load i32* %ptr1
  %c4 = zext i8 %c1 to i32
  %c5 = sext i16 %c2 to i32
  %c6 = add i32 %c4, %c5
  %c7 = sub i32 %c3, %c6
  store i32 %c7, i32* %ptr1
  ret void

; THUMB: ldrb
; THUMB: ldrh
; THUMB: uxtb
; THUMB: sxth
; THUMB: add
; THUMB: sub
; ARM: ldrb
; ARM: ldrh
; ARM: uxtb
; ARM: sxth
; ARM: add
; ARM: sub
}

; Check loads/stores with globals
@test4g = external global i32

define void @test4() {
  %a = load i32* @test4g
  %b = add i32 %a, 1
  store i32 %b, i32* @test4g
  ret void

; THUMB: ldr.n r0, LCPI4_1
; THUMB: ldr r0, [r0]
; THUMB: ldr r0, [r0]
; THUMB: adds r0, #1
; THUMB: ldr.n r1, LCPI4_0
; THUMB: ldr r1, [r1]
; THUMB: str r0, [r1]

; ARM: ldr r0, LCPI4_1
; ARM: ldr r0, [r0]
; ARM: ldr r0, [r0]
; ARM: add r0, r0, #1
; ARM: ldr r1, LCPI4_0
; ARM: ldr r1, [r1]
; ARM: str r0, [r1]
}

; Check unaligned stores
%struct.anon = type <{ float }>

@a = common global %struct.anon* null, align 4

define void @unaligned_store(float %x, float %y) nounwind {
entry:
; ARM: @unaligned_store
; ARM: vmov r1, s0
; ARM: str r1, [r0]

; THUMB: @unaligned_store
; THUMB: vmov r1, s0
; THUMB: str r1, [r0]

  %add = fadd float %x, %y
  %0 = load %struct.anon** @a, align 4
  %x1 = getelementptr inbounds %struct.anon* %0, i32 0, i32 0
  store float %add, float* %x1, align 1
  ret void
}
