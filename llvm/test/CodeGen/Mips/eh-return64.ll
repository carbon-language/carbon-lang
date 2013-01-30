; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s

declare void @llvm.eh.return.i64(i64, i8*)
declare void @foo(...)

define void @f1(i64 %offset, i8* %handler) {
entry:
  call void (...)* @foo()
  call void @llvm.eh.return.i64(i64 %offset, i8* %handler)
  unreachable

; CHECK:        f1
; CHECK:        daddiu  $sp, $sp, -[[spoffset:[0-9]+]]

; check that $a0-$a3 are saved on stack.
; CHECK:        sd      $4, [[offset0:[0-9]+]]($sp)
; CHECK:        sd      $5, [[offset1:[0-9]+]]($sp)
; CHECK:        sd      $6, [[offset2:[0-9]+]]($sp)
; CHECK:        sd      $7, [[offset3:[0-9]+]]($sp)

; check that .cfi_offset directives are emitted for $a0-$a3.
; CHECK:        .cfi_offset 4,
; CHECK:        .cfi_offset 5,
; CHECK:        .cfi_offset 6,
; CHECK:        .cfi_offset 7,

; check that stack adjustment and handler are put in $v1 and $v0.
; CHECK:        or      $[[R0:[a-z0-9]+]], $5, $zero
; CHECK:        or      $[[R1:[a-z0-9]+]], $4, $zero
; CHECK:        or      $3, $[[R1]], $zero
; CHECK:        or      $2, $[[R0]], $zero

; check that $a0-$a3 are restored from stack.
; CHECK:        ld      $4, [[offset0]]($sp)
; CHECK:        ld      $5, [[offset1]]($sp)
; CHECK:        ld      $6, [[offset2]]($sp)
; CHECK:        ld      $7, [[offset3]]($sp)

; check that stack is adjusted by $v1 and that code returns to address in $v0
; CHECK:        daddiu  $sp, $sp, [[spoffset]]
; CHECK:        or      $ra, $2, $zero
; CHECK:        jr      $ra
; CHECK:        daddu   $sp, $sp, $3

}

define void @f2(i64 %offset, i8* %handler) {
entry:
  call void @llvm.eh.return.i64(i64 %offset, i8* %handler)
  unreachable

; CHECK:        f2
; CHECK:        daddiu  $sp, $sp, -[[spoffset:[0-9]+]]

; check that $a0-$a3 are saved on stack.
; CHECK:        sd      $4, [[offset0:[0-9]+]]($sp)
; CHECK:        sd      $5, [[offset1:[0-9]+]]($sp)
; CHECK:        sd      $6, [[offset2:[0-9]+]]($sp)
; CHECK:        sd      $7, [[offset3:[0-9]+]]($sp)

; check that .cfi_offset directives are emitted for $a0-$a3.
; CHECK:        .cfi_offset 4,
; CHECK:        .cfi_offset 5,
; CHECK:        .cfi_offset 6,
; CHECK:        .cfi_offset 7,

; check that stack adjustment and handler are put in $v1 and $v0.
; CHECK:        or      $3, $4, $zero
; CHECK:        or      $2, $5, $zero

; check that $a0-$a3 are restored from stack.
; CHECK:        ld      $4, [[offset0]]($sp)
; CHECK:        ld      $5, [[offset1]]($sp)
; CHECK:        ld      $6, [[offset2]]($sp)
; CHECK:        ld      $7, [[offset3]]($sp)

; check that stack is adjusted by $v1 and that code returns to address in $v0
; CHECK:        daddiu  $sp, $sp, [[spoffset]]
; CHECK:        or      $ra, $2, $zero
; CHECK:        jr      $ra
; CHECK:        daddu   $sp, $sp, $3

}
