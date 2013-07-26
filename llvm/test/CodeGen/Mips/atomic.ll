; RUN: llc -march=mipsel --disable-machine-licm < %s | FileCheck %s -check-prefix=CHECK-EL
; RUN: llc -march=mips   --disable-machine-licm < %s | FileCheck %s -check-prefix=CHECK-EB

@x = common global i32 0, align 4

define i32 @AtomicLoadAdd32(i32 %incr) nounwind {
entry:
  %0 = atomicrmw add i32* @x, i32 %incr monotonic
  ret i32 %0

; CHECK-EL-LABEL:   AtomicLoadAdd32:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK-EL:   addu    $[[R2:[0-9]+]], $[[R1]], $4
; CHECK-EL:   sc      $[[R2]], 0($[[R0]])
; CHECK-EL:   beqz    $[[R2]], $[[BB0]]

; CHECK-EB-LABEL:   AtomicLoadAdd32:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK-EB:   addu    $[[R2:[0-9]+]], $[[R1]], $4
; CHECK-EB:   sc      $[[R2]], 0($[[R0]])
; CHECK-EB:   beqz    $[[R2]], $[[BB0]]
}

define i32 @AtomicLoadNand32(i32 %incr) nounwind {
entry:
  %0 = atomicrmw nand i32* @x, i32 %incr monotonic
  ret i32 %0

; CHECK-EL-LABEL:   AtomicLoadNand32:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK-EL:   and     $[[R3:[0-9]+]], $[[R1]], $4
; CHECK-EL:   nor     $[[R2:[0-9]+]], $zero, $[[R3]]
; CHECK-EL:   sc      $[[R2]], 0($[[R0]])
; CHECK-EL:   beqz    $[[R2]], $[[BB0]]

; CHECK-EB-LABEL:   AtomicLoadNand32:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK-EB:   and     $[[R3:[0-9]+]], $[[R1]], $4
; CHECK-EB:   nor     $[[R2:[0-9]+]], $zero, $[[R3]]
; CHECK-EB:   sc      $[[R2]], 0($[[R0]])
; CHECK-EB:   beqz    $[[R2]], $[[BB0]]
}

define i32 @AtomicSwap32(i32 %newval) nounwind {
entry:
  %newval.addr = alloca i32, align 4
  store i32 %newval, i32* %newval.addr, align 4
  %tmp = load i32* %newval.addr, align 4
  %0 = atomicrmw xchg i32* @x, i32 %tmp monotonic
  ret i32 %0

; CHECK-EL-LABEL:   AtomicSwap32:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      ${{[0-9]+}}, 0($[[R0]])
; CHECK-EL:   sc      $[[R2:[0-9]+]], 0($[[R0]])
; CHECK-EL:   beqz    $[[R2]], $[[BB0]]

; CHECK-EB-LABEL:   AtomicSwap32:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      ${{[0-9]+}}, 0($[[R0]])
; CHECK-EB:   sc      $[[R2:[0-9]+]], 0($[[R0]])
; CHECK-EB:   beqz    $[[R2]], $[[BB0]]
}

define i32 @AtomicCmpSwap32(i32 %oldval, i32 %newval) nounwind {
entry:
  %newval.addr = alloca i32, align 4
  store i32 %newval, i32* %newval.addr, align 4
  %tmp = load i32* %newval.addr, align 4
  %0 = cmpxchg i32* @x, i32 %oldval, i32 %tmp monotonic
  ret i32 %0

; CHECK-EL-LABEL:   AtomicCmpSwap32:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      $2, 0($[[R0]])
; CHECK-EL:   bne     $2, $4, $[[BB1:[A-Z_0-9]+]]
; CHECK-EL:   sc      $[[R2:[0-9]+]], 0($[[R0]])
; CHECK-EL:   beqz    $[[R2]], $[[BB0]]
; CHECK-EL:   $[[BB1]]:

; CHECK-EB-LABEL:   AtomicCmpSwap32:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      $2, 0($[[R0]])
; CHECK-EB:   bne     $2, $4, $[[BB1:[A-Z_0-9]+]]
; CHECK-EB:   sc      $[[R2:[0-9]+]], 0($[[R0]])
; CHECK-EB:   beqz    $[[R2]], $[[BB0]]
; CHECK-EB:   $[[BB1]]:
}



@y = common global i8 0, align 1

define signext i8 @AtomicLoadAdd8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw add i8* @y, i8 %incr monotonic
  ret i8 %0

; CHECK-EL-LABEL:   AtomicLoadAdd8:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EL:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EL:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EL:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EL:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK-EL:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK-EL:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK-EL:   sllv    $[[R9:[0-9]+]], $4, $[[R4]]

; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK-EL:   addu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK-EL:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK-EL:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK-EL:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK-EL:   sc      $[[R14]], 0($[[R2]])
; CHECK-EL:   beqz    $[[R14]], $[[BB0]]

; CHECK-EL:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK-EL:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK-EL:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EL:   sra     $2, $[[R17]], 24

; CHECK-EB-LABEL:   AtomicLoadAdd8:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EB:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EB:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EB:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EB:   xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:   sll     $[[R5:[0-9]+]], $[[R4]], 3
; CHECK-EB:   ori     $[[R6:[0-9]+]], $zero, 255
; CHECK-EB:   sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; CHECK-EB:   nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; CHECK-EB:   sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK-EB:   addu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK-EB:   and     $[[R12:[0-9]+]], $[[R11]], $[[R7]]
; CHECK-EB:   and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; CHECK-EB:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK-EB:   sc      $[[R14]], 0($[[R2]])
; CHECK-EB:   beqz    $[[R14]], $[[BB0]]

; CHECK-EB:   and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; CHECK-EB:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]
; CHECK-EB:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EB:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicLoadSub8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw sub i8* @y, i8 %incr monotonic
  ret i8 %0

; CHECK-EL-LABEL:   AtomicLoadSub8:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EL:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EL:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EL:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EL:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK-EL:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK-EL:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK-EL:   sllv     $[[R9:[0-9]+]], $4, $[[R4]]

; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK-EL:   subu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK-EL:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK-EL:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK-EL:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK-EL:   sc      $[[R14]], 0($[[R2]])
; CHECK-EL:   beqz    $[[R14]], $[[BB0]]

; CHECK-EL:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK-EL:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK-EL:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EL:   sra     $2, $[[R17]], 24

; CHECK-EB-LABEL:   AtomicLoadSub8:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EB:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EB:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EB:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EB:   xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:   sll     $[[R5:[0-9]+]], $[[R4]], 3
; CHECK-EB:   ori     $[[R6:[0-9]+]], $zero, 255
; CHECK-EB:   sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; CHECK-EB:   nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; CHECK-EB:   sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK-EB:   subu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK-EB:   and     $[[R12:[0-9]+]], $[[R11]], $[[R7]]
; CHECK-EB:   and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; CHECK-EB:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK-EB:   sc      $[[R14]], 0($[[R2]])
; CHECK-EB:   beqz    $[[R14]], $[[BB0]]

; CHECK-EB:   and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; CHECK-EB:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]
; CHECK-EB:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EB:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicLoadNand8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw nand i8* @y, i8 %incr monotonic
  ret i8 %0

; CHECK-EL-LABEL:   AtomicLoadNand8:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EL:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EL:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EL:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EL:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK-EL:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK-EL:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK-EL:   sllv    $[[R9:[0-9]+]], $4, $[[R4]]

; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK-EL:   and     $[[R18:[0-9]+]], $[[R10]], $[[R9]]
; CHECK-EL:   nor     $[[R11:[0-9]+]], $zero, $[[R18]]
; CHECK-EL:   and     $[[R12:[0-9]+]], $[[R11]], $[[R6]]
; CHECK-EL:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK-EL:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK-EL:   sc      $[[R14]], 0($[[R2]])
; CHECK-EL:   beqz    $[[R14]], $[[BB0]]

; CHECK-EL:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK-EL:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK-EL:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EL:   sra     $2, $[[R17]], 24

; CHECK-EB-LABEL:   AtomicLoadNand8:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EB:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EB:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EB:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EB:   xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:   sll     $[[R5:[0-9]+]], $[[R4]], 3
; CHECK-EB:   ori     $[[R6:[0-9]+]], $zero, 255
; CHECK-EB:   sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; CHECK-EB:   nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; CHECK-EB:   sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK-EB:   and     $[[R18:[0-9]+]], $[[R10]], $[[R9]]
; CHECK-EB:   nor     $[[R11:[0-9]+]], $zero, $[[R18]]
; CHECK-EB:   and     $[[R12:[0-9]+]], $[[R11]], $[[R7]]
; CHECK-EB:   and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; CHECK-EB:   or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; CHECK-EB:   sc      $[[R14]], 0($[[R2]])
; CHECK-EB:   beqz    $[[R14]], $[[BB0]]

; CHECK-EB:   and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; CHECK-EB:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]
; CHECK-EB:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EB:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicSwap8(i8 signext %newval) nounwind {
entry:
  %0 = atomicrmw xchg i8* @y, i8 %newval monotonic
  ret i8 %0

; CHECK-EL-LABEL:   AtomicSwap8:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EL:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EL:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EL:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EL:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK-EL:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK-EL:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK-EL:   sllv    $[[R9:[0-9]+]], $4, $[[R4]]

; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK-EL:   and     $[[R18:[0-9]+]], $[[R9]], $[[R6]]
; CHECK-EL:   and     $[[R13:[0-9]+]], $[[R10]], $[[R7]]
; CHECK-EL:   or      $[[R14:[0-9]+]], $[[R13]], $[[R18]]
; CHECK-EL:   sc      $[[R14]], 0($[[R2]])
; CHECK-EL:   beqz    $[[R14]], $[[BB0]]

; CHECK-EL:   and     $[[R15:[0-9]+]], $[[R10]], $[[R6]]
; CHECK-EL:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R4]]
; CHECK-EL:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EL:   sra     $2, $[[R17]], 24

; CHECK-EB-LABEL:   AtomicSwap8:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EB:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EB:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EB:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EB:   xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:   sll     $[[R5:[0-9]+]], $[[R4]], 3
; CHECK-EB:   ori     $[[R6:[0-9]+]], $zero, 255
; CHECK-EB:   sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; CHECK-EB:   nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; CHECK-EB:   sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      $[[R10:[0-9]+]], 0($[[R2]])
; CHECK-EB:   and     $[[R18:[0-9]+]], $[[R9]], $[[R7]]
; CHECK-EB:   and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; CHECK-EB:   or      $[[R14:[0-9]+]], $[[R13]], $[[R18]]
; CHECK-EB:   sc      $[[R14]], 0($[[R2]])
; CHECK-EB:   beqz    $[[R14]], $[[BB0]]

; CHECK-EB:   and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; CHECK-EB:   srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]
; CHECK-EB:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EB:   sra     $2, $[[R17]], 24
}

define signext i8 @AtomicCmpSwap8(i8 signext %oldval, i8 signext %newval) nounwind {
entry:
  %0 = cmpxchg i8* @y, i8 %oldval, i8 %newval monotonic
  ret i8 %0

; CHECK-EL-LABEL:   AtomicCmpSwap8:
; CHECK-EL:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EL:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EL:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EL:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:   sll     $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EL:   ori     $[[R5:[0-9]+]], $zero, 255
; CHECK-EL:   sllv    $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK-EL:   nor     $[[R7:[0-9]+]], $zero, $[[R6]]
; CHECK-EL:   andi    $[[R8:[0-9]+]], $4, 255
; CHECK-EL:   sllv    $[[R9:[0-9]+]], $[[R8]], $[[R4]]
; CHECK-EL:   andi    $[[R10:[0-9]+]], $5, 255
; CHECK-EL:   sllv    $[[R11:[0-9]+]], $[[R10]], $[[R4]]

; CHECK-EL:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EL:   ll      $[[R12:[0-9]+]], 0($[[R2]])
; CHECK-EL:   and     $[[R13:[0-9]+]], $[[R12]], $[[R6]]
; CHECK-EL:   bne     $[[R13]], $[[R9]], $[[BB1:[A-Z_0-9]+]]

; CHECK-EL:   and     $[[R14:[0-9]+]], $[[R12]], $[[R7]]
; CHECK-EL:   or      $[[R15:[0-9]+]], $[[R14]], $[[R11]]
; CHECK-EL:   sc      $[[R15]], 0($[[R2]])
; CHECK-EL:   beqz    $[[R15]], $[[BB0]]

; CHECK-EL:   $[[BB1]]:
; CHECK-EL:   srlv    $[[R16:[0-9]+]], $[[R13]], $[[R4]]
; CHECK-EL:   sll     $[[R17:[0-9]+]], $[[R16]], 24
; CHECK-EL:   sra     $2, $[[R17]], 24

; CHECK-EB-LABEL:   AtomicCmpSwap8:
; CHECK-EB:   lw      $[[R0:[0-9]+]], %got(y)
; CHECK-EB:   addiu   $[[R1:[0-9]+]], $zero, -4
; CHECK-EB:   and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; CHECK-EB:   andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EB:   xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:   sll     $[[R5:[0-9]+]], $[[R4]], 3
; CHECK-EB:   ori     $[[R6:[0-9]+]], $zero, 255
; CHECK-EB:   sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; CHECK-EB:   nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; CHECK-EB:   andi    $[[R9:[0-9]+]], $4, 255
; CHECK-EB:   sllv    $[[R10:[0-9]+]], $[[R9]], $[[R5]]
; CHECK-EB:   andi    $[[R11:[0-9]+]], $5, 255
; CHECK-EB:   sllv    $[[R12:[0-9]+]], $[[R11]], $[[R5]]

; CHECK-EB:   $[[BB0:[A-Z_0-9]+]]:
; CHECK-EB:   ll      $[[R13:[0-9]+]], 0($[[R2]])
; CHECK-EB:   and     $[[R14:[0-9]+]], $[[R13]], $[[R7]]
; CHECK-EB:   bne     $[[R14]], $[[R10]], $[[BB1:[A-Z_0-9]+]]

; CHECK-EB:   and     $[[R15:[0-9]+]], $[[R13]], $[[R8]]
; CHECK-EB:   or      $[[R16:[0-9]+]], $[[R15]], $[[R12]]
; CHECK-EB:   sc      $[[R16]], 0($[[R2]])
; CHECK-EB:   beqz    $[[R16]], $[[BB0]]

; CHECK-EB:   $[[BB1]]:
; CHECK-EB:   srlv    $[[R17:[0-9]+]], $[[R14]], $[[R5]]
; CHECK-EB:   sll     $[[R18:[0-9]+]], $[[R17]], 24
; CHECK-EB:   sra     $2, $[[R18]], 24
}

@countsint = common global i32 0, align 4

define i32 @CheckSync(i32 %v) nounwind noinline {
entry:
  %0 = atomicrmw add i32* @countsint, i32 %v seq_cst
  ret i32 %0 

; CHECK-EL-LABEL:   CheckSync:
; CHECK-EL:   sync 0
; CHECK-EL:   ll
; CHECK-EL:   sc
; CHECK-EL:   beq
; CHECK-EL:   sync 0

; CHECK-EB-LABEL:   CheckSync:
; CHECK-EB:   sync 0
; CHECK-EB:   ll
; CHECK-EB:   sc
; CHECK-EB:   beq
; CHECK-EB:   sync 0
}

; make sure that this assertion in
; TwoAddressInstructionPass::TryInstructionTransform does not fail:
;
; line 1203: assert(TargetRegisterInfo::isVirtualRegister(regB) &&
;
; it failed when MipsDAGToDAGISel::ReplaceUsesWithZeroReg replaced an
; operand of an atomic instruction with register $zero. 
@a = external global i32

define i32 @zeroreg() nounwind {
entry:
  %0 = cmpxchg i32* @a, i32 1, i32 0 seq_cst
  %1 = icmp eq i32 %0, 1
  %conv = zext i1 %1 to i32
  ret i32 %conv
}
