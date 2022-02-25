; RUN: llc -mtriple=thumbv7m-linux-gnu < %s | FileCheck %s --check-prefix=CHECK --check-prefix=T2
; RUN: llc -mtriple=thumbv6m-linux-gnu < %s | FileCheck %s --check-prefix=CHECK --check-prefix=T1

; CHECK-LABEL: single_bit:
; CHECK: lsls r0, r0, #23
; T2-NEXT: mov
; T2-NEXT: it
; T1-NEXT: bmi
define i32 @single_bit(i32 %p) {
  %a = and i32 %p, 256
  %b = icmp eq i32 %a, 0
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

; CHECK-LABEL: single_bit_multi_use:
; CHECK: lsls r0, r0, #23
; T2-NEXT: mov
; T2-NEXT: it
; T1-NEXT: bmi
define i32 @single_bit_multi_use(i32 %p, i32* %z) {
  store i32 %p, i32* %z
  %a = and i32 %p, 256
  %b = icmp eq i32 %a, 0
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

; CHECK-LABEL: multi_bit_lsb_ubfx:
; CHECK: lsls r0, r0, #24
; T2-NEXT: mov
; T2-NEXT: it
; T1-NEXT: beq
define i32 @multi_bit_lsb_ubfx(i32 %p) {
  %a = and i32 %p, 255
  %b = icmp eq i32 %a, 0
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

; CHECK-LABEL: multi_bit_msb:
; CHECK: lsrs r0, r0, #24
; T2-NEXT: mov
; T2-NEXT: it
; T1-NEXT: beq
define i32 @multi_bit_msb(i32 %p) {
  %a = and i32 %p, 4278190080  ; 0xff000000
  %b = icmp eq i32 %a, 0
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

; CHECK-LABEL: multi_bit_nosb:
; T1: lsls r0, r0, #8
; T1-NEXT: lsrs r0, r0, #24
; T2: tst.w
; T2-NEXT: it
; T1-NEXT: beq
define i32 @multi_bit_nosb(i32 %p) {
  %a = and i32 %p, 16711680 ; 0x00ff0000
  %b = icmp eq i32 %a, 0
  br i1 %b, label %true, label %false

true:
  ret i32 1

false:
  ret i32 2
}

; CHECK-LABEL: i16_cmpz:
; T1:      uxth    r0, r0
; T1-NEXT: lsrs    r0, r0, #9
; T1-NEXT: bne
; T2:      uxth    r0, r0
; T2-NEXT: movs    r2, #0
; T2-NEXT: cmp.w   r2, r0, lsr #9
define void @i16_cmpz(i16 %x, void (i32)* %foo) {
entry:
  %cmp = icmp ult i16 %x, 512
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void %foo(i32 0) #1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
