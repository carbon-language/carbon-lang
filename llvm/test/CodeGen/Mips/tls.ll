; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=PIC
; RUN: llc -march=mipsel -relocation-model=static < %s \
; RUN:                             | FileCheck %s -check-prefix=STATIC


@t1 = thread_local global i32 0, align 4

define i32 @f1() nounwind {
entry:
  %tmp = load i32* @t1, align 4
  ret i32 %tmp

; CHECK: f1:

; PIC:   lw      $25, %call16(__tls_get_addr)($gp)
; PIC:   addiu   $4, $gp, %tlsgd(t1)
; PIC:   jalr    $25
; PIC:   lw      $2, 0($2)

; STATIC:   rdhwr   $3, $29
; STATIC:   lui     $[[R0:[0-9]+]], %tprel_hi(t1)
; STATIC:   addiu   $[[R1:[0-9]+]], $[[R0]], %tprel_lo(t1)
; STATIC:   addu    $[[R2:[0-9]+]], $3, $[[R1]]
; STATIC:   lw      $2, 0($[[R2]])
}


@t2 = external thread_local global i32

define i32 @f2() nounwind {
entry:
  %tmp = load i32* @t2, align 4
  ret i32 %tmp

; CHECK: f2:

; PIC:   lw      $25, %call16(__tls_get_addr)($gp)
; PIC:   addiu   $4, $gp, %tlsgd(t2)
; PIC:   jalr    $25
; PIC:   lw      $2, 0($2)

; STATIC:   rdhwr   $3, $29
; STATIC:   lw      $[[R0:[0-9]+]], %gottprel(t2)($gp)
; STATIC:   addu    $[[R1:[0-9]+]], $3, $[[R0]]
; STATIC:   lw      $2, 0($[[R1]])
}
