; RUN: llc < %s -march=arm                  | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -march=arm -mattr=+thumb2   | FileCheck %s --check-prefix=ARMT2
; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s --check-prefix=THUMB2

define i32 @t1(i32 %c) nounwind readnone {
entry:
; ARM-LABEL: t1:
; ARM: mov [[R1:r[0-9]+]], #101
; ARM: orr [[R1b:r[0-9]+]], [[R1]], #256
; ARM: movgt {{r[0-1]}}, #123

; ARMT2-LABEL: t1:
; ARMT2: movw [[R:r[0-1]]], #357
; ARMT2: movwgt [[R]], #123

; THUMB2-LABEL: t1:
; THUMB2: movw [[R:r[0-1]]], #357
; THUMB2: movgt [[R]], #123

  %0 = icmp sgt i32 %c, 1
  %1 = select i1 %0, i32 123, i32 357
  ret i32 %1
}

define i32 @t2(i32 %c) nounwind readnone {
entry:
; ARM-LABEL: t2:
; ARM: mov [[R:r[0-9]+]], #101
; ARM: orr [[R]], [[R]], #256
; ARM: movle [[R]], #123

; ARMT2-LABEL: t2:
; ARMT2: mov [[R:r[0-1]]], #123
; ARMT2: movwgt [[R]], #357

; THUMB2-LABEL: t2:
; THUMB2: mov{{(s|\.w)}} [[R:r[0-1]]], #123
; THUMB2: movwgt [[R]], #357

  %0 = icmp sgt i32 %c, 1
  %1 = select i1 %0, i32 357, i32 123
  ret i32 %1
}

define i32 @t3(i32 %a) nounwind readnone {
entry:
; ARM-LABEL: t3:
; ARM: mov [[R:r[0-1]]], #0
; ARM: moveq [[R]], #1

; ARMT2-LABEL: t3:
; ARMT2: mov [[R:r[0-1]]], #0
; ARMT2: movweq [[R]], #1

; THUMB2-LABEL: t3:
; THUMB2: mov{{(s|\.w)}} [[R:r[0-1]]], #0
; THUMB2: moveq [[R]], #1
  %0 = icmp eq i32 %a, 160
  %1 = zext i1 %0 to i32
  ret i32 %1
}

define i32 @t4(i32 %a, i32 %b, i32 %x) nounwind {
entry:
; ARM-LABEL: t4:
; ARM: ldr
; ARM: mov{{lt|ge}}

; ARMT2-LABEL: t4:
; ARMT2: movwlt [[R0:r[0-9]+]], #65365
; ARMT2: movtlt [[R0]], #65365

; THUMB2-LABEL: t4:
; THUMB2: mvnlt [[R0:r[0-9]+]], #11141290
  %0 = icmp slt i32 %a, %b
  %1 = select i1 %0, i32 4283826005, i32 %x
  ret i32 %1
}

; rdar://9758317
define i32 @t5(i32 %a) nounwind {
entry:
; ARM-LABEL: t5:
; ARM-NOT: mov
; ARM: cmp r0, #1
; ARM-NOT: mov
; ARM: movne r0, #0

; THUMB2-LABEL: t5:
; THUMB2-NOT: mov
; THUMB2: cmp r0, #1
; THUMB2: it ne
; THUMB2: movne r0, #0
  %cmp = icmp eq i32 %a, 1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @t6(i32 %a) nounwind {
entry:
; ARM-LABEL: t6:
; ARM-NOT: mov
; ARM: cmp r0, #0
; ARM: movne r0, #1

; THUMB2-LABEL: t6:
; THUMB2-NOT: mov
; THUMB2: cmp r0, #0
; THUMB2: it ne
; THUMB2: movne r0, #1
  %tobool = icmp ne i32 %a, 0
  %lnot.ext = zext i1 %tobool to i32
  ret i32 %lnot.ext
}
