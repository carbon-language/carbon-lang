; RUN: llc -march=mipsel -mcpu=mips2 < %s | FileCheck %s


declare i32 @llvm.atomic.load.add.i32.p0i32(i32* nocapture, i32) nounwind
declare i32 @llvm.atomic.load.nand.i32.p0i32(i32* nocapture, i32) nounwind
declare i32 @llvm.atomic.swap.i32.p0i32(i32* nocapture, i32) nounwind
declare i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* nocapture, i32, i32) nounwind

declare i8 @llvm.atomic.load.add.i8.p0i8(i8* nocapture, i8) nounwind
declare i8 @llvm.atomic.load.sub.i8.p0i8(i8* nocapture, i8) nounwind
declare i8 @llvm.atomic.load.nand.i8.p0i8(i8* nocapture, i8) nounwind
declare i8 @llvm.atomic.swap.i8.p0i8(i8* nocapture, i8) nounwind
declare i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* nocapture, i8, i8) nounwind


@x = common global i32 0, align 4

define i32 @AtomicLoadAdd32(i32 %incr) nounwind {
entry:
  %0 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* @x, i32 %incr)
  ret i32 %0

; CHECK:   AtomicLoadAdd32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)($gp)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK:   or      $2, $zero, $[[R1]]
; CHECK:   addu    $[[R2:[0-9]+]], $[[R1]], $4
; CHECK:   sc      $[[R2]], 0($[[R0]])
; CHECK:   beq     $[[R2]], $zero, $[[BB0]]
}

define i32 @AtomicLoadNand32(i32 %incr) nounwind {
entry:
  %0 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* @x, i32 %incr)
  ret i32 %0

; CHECK:   AtomicLoadNand32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)($gp)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK:   or      $2, $zero, $[[R1]]
; CHECK:   and     $[[R1]], $[[R1]], $4
; CHECK:   nor     $[[R2:[0-9]+]], $zero, $[[R1]]
; CHECK:   sc      $[[R2]], 0($[[R0]])
; CHECK:   beq     $[[R2]], $zero, $[[BB0]]
}

define i32 @AtomicSwap32(i32 %oldval) nounwind {
entry:
  %0 = call i32 @llvm.atomic.swap.i32.p0i32(i32* @x, i32 %oldval)
  ret i32 %0

; CHECK:   AtomicSwap32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)($gp)
; CHECK:   sw      $4, [[OFFSET:[0-9]+]]($sp)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK:   or      $2, $zero, $[[R1]]
; CHECK:   lw      $[[R2:[0-9]+]], [[OFFSET]]($sp)
; CHECK:   or      $[[R3:[0-9]+]], $zero, $[[R2]]
; CHECK:   sc      $[[R3]], 0($[[R0]])
; CHECK:   beq     $[[R3]], $zero, $[[BB0]]
}

define i32 @AtomicCmpSwap32(i32 %oldval, i32 %newval) nounwind {
entry:
  %0 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* @x, i32 %oldval, i32 %newval)
  ret i32 %0

; CHECK:   AtomicCmpSwap32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)($gp)
; CHECK:   sw      $5, [[OFFSET:[0-9]+]]($sp)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $2, 0($[[R0]])
; CHECK:   bne     $2, $4, $[[BB1:[A-Z_0-9]+]]
; CHECK:   lw      $[[R1:[0-9]+]], [[OFFSET]]($sp)
; CHECK:   or      $[[R2:[0-9]+]], $zero, $[[R1]]
; CHECK:   sc      $[[R2]], 0($[[R0]])
; CHECK:   beq     $[[R2]], $zero, $[[BB0]]
; CHECK:   $[[BB1]]:
}



@y = common global i8 0, align 1

define signext i8 @AtomicLoadAdd8(i8 signext %incr) nounwind {
entry:
  %0 = call i8 @llvm.atomic.load.add.i8.p0i8(i8* @y, i8 %incr)
  ret i8 %0

; CHECK:   AtomicLoadAdd8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)($gp)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sll     $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   andi    $[[R8:[0-9]+]], $4, 255
; CHECK:   sll     $[[R9:[0-9]+]], $[[R8]], $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK:   addu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK:   sc      $[[R14]], 0($[[R2]])
; CHECK:   beq     $[[R14]], $zero, $[[BB0]]

; CHECK:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK:   srl     $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicLoadSub8(i8 signext %incr) nounwind {
entry:
  %0 = call i8 @llvm.atomic.load.sub.i8.p0i8(i8* @y, i8 %incr)
  ret i8 %0

; CHECK:   AtomicLoadSub8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)($gp)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sll     $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   subu    $[[R18:[0-9]+]], $zero, $4
; CHECK:   andi    $[[R8:[0-9]+]], $[[R18]], 255
; CHECK:   sll     $[[R9:[0-9]+]], $[[R8]], $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK:   addu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK:   sc      $[[R14]], 0($[[R2]])
; CHECK:   beq     $[[R14]], $zero, $[[BB0]]

; CHECK:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK:   srl     $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicLoadNand8(i8 signext %incr) nounwind {
entry:
  %0 = call i8 @llvm.atomic.load.nand.i8.p0i8(i8* @y, i8 %incr)
  ret i8 %0

; CHECK:   AtomicLoadNand8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)($gp)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sll     $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   andi    $[[R8:[0-9]+]], $4, 255
; CHECK:   sll     $[[R9:[0-9]+]], $[[R8]], $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK:   and     $[[R18:[0-9]+]], $[[R10]], $[[R9]]
; CHECK:   nor     $[[R11:[0-9]+]], $zero, $[[R18]]
; CHECK:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK:   sc      $[[R14]], 0($[[R2]])
; CHECK:   beq     $[[R14]], $zero, $[[BB0]]

; CHECK:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK:   srl     $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicSwap8(i8 signext %oldval) nounwind {
entry:
  %0 = call i8 @llvm.atomic.swap.i8.p0i8(i8* @y, i8 %oldval)
  ret i8 %0

; CHECK:   AtomicSwap8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)($gp)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sll     $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   andi    $[[R8:[0-9]+]], $4, 255
; CHECK:   sll     $[[R9:[0-9]+]], $[[R8]], $[[R4]]
; CHECK:   sw      $[[R9]], [[OFFSET:[0-9]+]]($sp)

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK:   lw      $[[R18:[0-9]+]], [[OFFSET]]($sp)
; CHECK:   or      $[[R11:[0-9]+]], $zero, $[[R18]]
; CHECK:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK:   sc      $[[R14]], 0($[[R2]])
; CHECK:   beq     $[[R14]], $zero, $[[BB0]]

; CHECK:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK:   srl     $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicCmpSwap8(i8 signext %oldval, i8 signext %newval) nounwind {
entry:
  %0 = call i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* @y, i8 %oldval, i8 %newval)
  ret i8 %0

; CHECK:   AtomicCmpSwap8:
; CHECK:   lw      $[[R0:[0-9]+]], %got(y)($gp)
; CHECK:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK:   sll     $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK:   andi    $[[R8:[0-9]+]], $4, 255
; CHECK:   sll     $[[R9:[0-9]+]], $[[R8]], $[[R4]]
; CHECK:   andi    $[[R10:[0-9]+]], $5, 255
; CHECK:   sll     $[[R11:[0-9]+]], $[[R10]], $[[R4]]

; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R12:[0-9]+]], 0($[[R2]])
; CHECK:   and     $[[R13:[0-9]+]], $[[R12]], $[[R6]]
; CHECK:   bne     $[[R13]], $[[R9]], $[[BB1:[A-Z_0-9]+]]

; CHECK:   and     $[[R14:[0-9]+]], $[[R12]], $[[R7]]
; CHECK:   or      $[[R15:[0-9]+]], $[[R14]], $[[R11]]
; CHECK:   sc      $[[R15]], 0($[[R2]])
; CHECK:   beq     $[[R15]], $zero, $[[BB0]]

; CHECK:   $[[BB1]]:
; CHECK:   srl     $[[R16:[0-9]+]], $[[R13]], $[[R4]]
; CHECK:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK:   sra     $2, $[[R17]], 24
}
