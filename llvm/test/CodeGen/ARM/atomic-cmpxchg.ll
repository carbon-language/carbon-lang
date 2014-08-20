; RUN: llc < %s -mtriple=arm-linux-gnueabi -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-ARM
; RUN: llc < %s -mtriple=thumb-linux-gnueabi -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-THUMB

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-ARMV7
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-THUMBV7

define zeroext i1 @test_cmpxchg_res_i8(i8* %addr, i8 %desired, i8 zeroext %new) {
entry:
  %0 = cmpxchg i8* %addr, i8 %desired, i8 %new monotonic monotonic
  %1 = extractvalue { i8, i1 } %0, 1
  ret i1 %1
}

; CHECK-ARM-LABEL: test_cmpxchg_res_i8
; CHECK-ARM: bl __sync_val_compare_and_swap_1
; CHECK-ARM: mov [[REG:r[0-9]+]], #0
; CHECK-ARM: cmp r0, {{r[0-9]+}}
; CHECK-ARM: moveq [[REG]], #1
; CHECK-ARM: mov r0, [[REG]]

; CHECK-THUMB-LABEL: test_cmpxchg_res_i8
; CHECK-THUMB: bl __sync_val_compare_and_swap_1
; CHECK-THUMB-NOT: mov [[R1:r[0-7]]], r0
; CHECK-THUMB: push  {r0}
; CHECK-THUMB: pop {[[R1:r[0-7]]]}
; CHECK-THUMB: movs r0, #1
; CHECK-THUMB: movs [[R2:r[0-9]+]], #0
; CHECK-THUMB: cmp [[R1]], {{r[0-9]+}}
; CHECK-THU<B: beq
; CHECK-THUMB: push  {[[R2]]}
; CHECK-THUMB: pop {r0}

; CHECK-ARMV7-LABEL: test_cmpxchg_res_i8
; CHECK-ARMV7: ldrexb [[R3:r[0-9]+]], [r0]
; CHECK-ARMV7: mov [[R1:r[0-9]+]], #0
; CHECK-ARMV7: cmp [[R3]], {{r[0-9]+}}
; CHECK-ARMV7: bne
; CHECK-ARMV7: strexb [[R3]], {{r[0-9]+}}, [{{r[0-9]+}}]
; CHECK-ARMV7: mov [[R1]], #1
; CHECK-ARMV7: cmp [[R3]], #0
; CHECK-ARMV7: bne
; CHECK-ARMV7: mov r0, [[R1]]

; CHECK-THUMBV7-LABEL: test_cmpxchg_res_i8
; CHECK-THUMBV7: ldrexb [[R3:r[0-9]+]], [r0]
; CHECK-THUMBV7: cmp [[R3]], {{r[0-9]+}}
; CHECK-THUMBV7: movne r0, #0
; CHECK-THUMBV7: bxne lr
; CHECK-THUMBV7: strexb [[R3]], {{r[0-9]+}}, [{{r[0-9]+}}]
; CHECK-THUMBV7: cmp [[R3]], #0
; CHECK-THUMBV7: itt eq
; CHECK-THUMBV7: moveq r0, #1
; CHECK-THUMBV7: bxeq lr
