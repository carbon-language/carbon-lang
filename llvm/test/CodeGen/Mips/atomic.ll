; RUN: llc -march=mipsel --disable-machine-licm < %s | FileCheck %s -check-prefix=ALL -check-prefix=CHECK-EL
; RUN: llc -march=mips   --disable-machine-licm < %s | FileCheck %s -check-prefix=ALL -check-prefix=CHECK-EB

@x = common global i32 0, align 4

define i32 @AtomicLoadAdd32(i32 %incr) nounwind {
entry:
  %0 = atomicrmw add i32* @x, i32 %incr monotonic
  ret i32 %0

; ALL-LABEL: AtomicLoadAdd32:

; ALL:           lw      $[[R0:[0-9]+]], %got(x)
; ALL:       $[[BB0:[A-Z_0-9]+]]:
; ALL:           ll      $[[R1:[0-9]+]], 0($[[R0]])
; ALL:           addu    $[[R2:[0-9]+]], $[[R1]], $4
; ALL:           sc      $[[R2]], 0($[[R0]])
; ALL:           beqz    $[[R2]], $[[BB0]]
}

define i32 @AtomicLoadNand32(i32 %incr) nounwind {
entry:
  %0 = atomicrmw nand i32* @x, i32 %incr monotonic
  ret i32 %0

; ALL-LABEL: AtomicLoadNand32:

; ALL:           lw      $[[R0:[0-9]+]], %got(x)
; ALL:       $[[BB0:[A-Z_0-9]+]]:
; ALL:           ll      $[[R1:[0-9]+]], 0($[[R0]])
; ALL:           and     $[[R3:[0-9]+]], $[[R1]], $4
; ALL:           nor     $[[R2:[0-9]+]], $zero, $[[R3]]
; ALL:           sc      $[[R2]], 0($[[R0]])
; ALL:           beqz    $[[R2]], $[[BB0]]
}

define i32 @AtomicSwap32(i32 %newval) nounwind {
entry:
  %newval.addr = alloca i32, align 4
  store i32 %newval, i32* %newval.addr, align 4
  %tmp = load i32* %newval.addr, align 4
  %0 = atomicrmw xchg i32* @x, i32 %tmp monotonic
  ret i32 %0

; ALL-LABEL: AtomicSwap32:

; ALL:           lw      $[[R0:[0-9]+]], %got(x)
; ALL:       $[[BB0:[A-Z_0-9]+]]:
; ALL:           ll      ${{[0-9]+}}, 0($[[R0]])
; ALL:           sc      $[[R2:[0-9]+]], 0($[[R0]])
; ALL:           beqz    $[[R2]], $[[BB0]]
}

define i32 @AtomicCmpSwap32(i32 %oldval, i32 %newval) nounwind {
entry:
  %newval.addr = alloca i32, align 4
  store i32 %newval, i32* %newval.addr, align 4
  %tmp = load i32* %newval.addr, align 4
  %0 = cmpxchg i32* @x, i32 %oldval, i32 %tmp monotonic monotonic
  %1 = extractvalue { i32, i1 } %0, 0
  ret i32 %1

; ALL-LABEL: AtomicCmpSwap32:

; ALL:           lw      $[[R0:[0-9]+]], %got(x)
; ALL:       $[[BB0:[A-Z_0-9]+]]:
; ALL:           ll      $2, 0($[[R0]])
; ALL:           bne     $2, $4, $[[BB1:[A-Z_0-9]+]]
; ALL:           sc      $[[R2:[0-9]+]], 0($[[R0]])
; ALL:           beqz    $[[R2]], $[[BB0]]
; ALL:       $[[BB1]]:
}



@y = common global i8 0, align 1

define signext i8 @AtomicLoadAdd8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw add i8* @y, i8 %incr monotonic
  ret i8 %0

; ALL-LABEL: AtomicLoadAdd8:

; ALL:           lw      $[[R0:[0-9]+]], %got(y)
; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; ALL:       $[[BB0:[A-Z_0-9]+]]:
; ALL:           ll      $[[R10:[0-9]+]], 0($[[R2]])
; ALL:           addu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; ALL:           and     $[[R12:[0-9]+]], $[[R11]], $[[R7]]
; ALL:           and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; ALL:           or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; ALL:           sc      $[[R14]], 0($[[R2]])
; ALL:           beqz    $[[R14]], $[[BB0]]

; ALL:           and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; ALL:           srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]
; ALL:           sll     $[[R17:[0-9]+]], $[[R16]], 24
; ALL:           sra     $2, $[[R17]], 24
}

define signext i8 @AtomicLoadSub8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw sub i8* @y, i8 %incr monotonic
  ret i8 %0

; ALL-LABEL: AtomicLoadSub8:

; ALL:        lw      $[[R0:[0-9]+]], %got(y)
; ALL:        addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:        and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:        andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:   sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:   xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:   sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:        ori     $[[R6:[0-9]+]], $zero, 255
; ALL:        sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:        nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:        sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; ALL:    $[[BB0:[A-Z_0-9]+]]:
; ALL:        ll      $[[R10:[0-9]+]], 0($[[R2]])
; ALL:        subu    $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; ALL:        and     $[[R12:[0-9]+]], $[[R11]], $[[R7]]
; ALL:        and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; ALL:        or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; ALL:        sc      $[[R14]], 0($[[R2]])
; ALL:        beqz    $[[R14]], $[[BB0]]

; ALL:        and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; ALL:        srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]
; ALL:        sll     $[[R17:[0-9]+]], $[[R16]], 24
; ALL:        sra     $2, $[[R17]], 24
}

define signext i8 @AtomicLoadNand8(i8 signext %incr) nounwind {
entry:
  %0 = atomicrmw nand i8* @y, i8 %incr monotonic
  ret i8 %0

; ALL-LABEL: AtomicLoadNand8:

; ALL:           lw      $[[R0:[0-9]+]], %got(y)
; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; ALL:       $[[BB0:[A-Z_0-9]+]]:
; ALL:           ll      $[[R10:[0-9]+]], 0($[[R2]])
; ALL:           and     $[[R18:[0-9]+]], $[[R10]], $[[R9]]
; ALL:           nor     $[[R11:[0-9]+]], $zero, $[[R18]]
; ALL:           and     $[[R12:[0-9]+]], $[[R11]], $[[R7]]
; ALL:           and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; ALL:           or      $[[R14:[0-9]+]], $[[R13]], $[[R12]]
; ALL:           sc      $[[R14]], 0($[[R2]])
; ALL:           beqz    $[[R14]], $[[BB0]]

; ALL:           and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; ALL:           srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]
; ALL:           sll     $[[R17:[0-9]+]], $[[R16]], 24
; ALL:           sra     $2, $[[R17]], 24
}

define signext i8 @AtomicSwap8(i8 signext %newval) nounwind {
entry:
  %0 = atomicrmw xchg i8* @y, i8 %newval monotonic
  ret i8 %0

; ALL-LABEL: AtomicSwap8:

; ALL:           lw      $[[R0:[0-9]+]], %got(y)
; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           sllv    $[[R9:[0-9]+]], $4, $[[R5]]

; ALL:       $[[BB0:[A-Z_0-9]+]]:
; ALL:           ll      $[[R10:[0-9]+]], 0($[[R2]])
; ALL:           and     $[[R18:[0-9]+]], $[[R9]], $[[R7]]
; ALL:           and     $[[R13:[0-9]+]], $[[R10]], $[[R8]]
; ALL:           or      $[[R14:[0-9]+]], $[[R13]], $[[R18]]
; ALL:           sc      $[[R14]], 0($[[R2]])
; ALL:           beqz    $[[R14]], $[[BB0]]

; ALL:           and     $[[R15:[0-9]+]], $[[R10]], $[[R7]]
; ALL:           srlv    $[[R16:[0-9]+]], $[[R15]], $[[R5]]
; ALL:           sll     $[[R17:[0-9]+]], $[[R16]], 24
; ALL:           sra     $2, $[[R17]], 24
}

define signext i8 @AtomicCmpSwap8(i8 signext %oldval, i8 signext %newval) nounwind {
entry:
  %pair0 = cmpxchg i8* @y, i8 %oldval, i8 %newval monotonic monotonic
  %0 = extractvalue { i8, i1 } %pair0, 0
  ret i8 %0

; ALL-LABEL: AtomicCmpSwap8:

; ALL:           lw      $[[R0:[0-9]+]], %got(y)
; ALL:           addiu   $[[R1:[0-9]+]], $zero, -4
; ALL:           and     $[[R2:[0-9]+]], $[[R0]], $[[R1]]
; ALL:           andi    $[[R3:[0-9]+]], $[[R0]], 3
; CHECK-EL:      sll     $[[R5:[0-9]+]], $[[R3]], 3
; CHECK-EB:      xori    $[[R4:[0-9]+]], $[[R3]], 3
; CHECK-EB:      sll     $[[R5:[0-9]+]], $[[R4]], 3
; ALL:           ori     $[[R6:[0-9]+]], $zero, 255
; ALL:           sllv    $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; ALL:           nor     $[[R8:[0-9]+]], $zero, $[[R7]]
; ALL:           andi    $[[R9:[0-9]+]], $4, 255
; ALL:           sllv    $[[R10:[0-9]+]], $[[R9]], $[[R5]]
; ALL:           andi    $[[R11:[0-9]+]], $5, 255
; ALL:           sllv    $[[R12:[0-9]+]], $[[R11]], $[[R5]]

; ALL:       $[[BB0:[A-Z_0-9]+]]:
; ALL:           ll      $[[R13:[0-9]+]], 0($[[R2]])
; ALL:           and     $[[R14:[0-9]+]], $[[R13]], $[[R7]]
; ALL:           bne     $[[R14]], $[[R10]], $[[BB1:[A-Z_0-9]+]]

; ALL:           and     $[[R15:[0-9]+]], $[[R13]], $[[R8]]
; ALL:           or      $[[R16:[0-9]+]], $[[R15]], $[[R12]]
; ALL:           sc      $[[R16]], 0($[[R2]])
; ALL:           beqz    $[[R16]], $[[BB0]]

; ALL:       $[[BB1]]:
; ALL:           srlv    $[[R17:[0-9]+]], $[[R14]], $[[R5]]
; ALL:           sll     $[[R18:[0-9]+]], $[[R17]], 24
; ALL:           sra     $2, $[[R18]], 24
}

@countsint = common global i32 0, align 4

define i32 @CheckSync(i32 %v) nounwind noinline {
entry:
  %0 = atomicrmw add i32* @countsint, i32 %v seq_cst
  ret i32 %0 

; ALL-LABEL: CheckSync:

; ALL:           sync 0
; ALL:           ll
; ALL:           sc
; ALL:           beq
; ALL:           sync 0
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
  %pair0 = cmpxchg i32* @a, i32 1, i32 0 seq_cst seq_cst
  %0 = extractvalue { i32, i1 } %pair0, 0
  %1 = icmp eq i32 %0, 1
  %conv = zext i1 %1 to i32
  ret i32 %conv
}
