; RUN: llc -march=arm -mattr=+v4t < %s | FileCheck -check-prefix PRE6 %s
; RUN: llc -march=armeb -mattr=+v4t < %s | FileCheck -check-prefix PRE6BE %s
; RUN: llc -march=arm -mattr=+v5t < %s | FileCheck -check-prefix PRE6 %s
; RUN: llc -march=arm -mattr=+v5te < %s | FileCheck -check-prefix PRE6 %s
; RUN: llc -march=arm -mattr=+v6 < %s | FileCheck -check-prefix REV %s
; RUN: llc -march=armeb -mattr=+v6 < %s | FileCheck -check-prefix REVBE %s
; RUN: llc -march=arm -mattr=+v6k < %s | FileCheck -check-prefix REV %s
; RUN: llc -march=arm -mattr=+v6m < %s | FileCheck -check-prefix REV %s
; RUN: llc -march=arm -mattr=+v6t2 < %s | FileCheck -check-prefix REV %s
; RUN: llc -march=arm -mattr=+v7 < %s | FileCheck -check-prefix REV %s
; RUN: llc -march=arm -mattr=+v8 < %s | FileCheck -check-prefix REV %s
; RUN: llc -march=arm -mattr=+v8.1a < %s | FileCheck -check-prefix REV %s

;; Test byte swap instrinsic lowering on ARM targets.

;; The REV instruction only appeared in ARMv6 and later. Earlier
;; supported architecture have to open-code this intrinsic.

define i16 @bswap16(i16 %x) #0 {
  %1 = tail call i16 @llvm.bswap.i16(i16 %x)
  ret i16 %1
; PRE6-LABEL: bswap16
;; The source register patterns are all capable of matching a new
;; register to avoid specifying allocation choices unnecessarily.
; PRE6:       mov [[R1:r[0-9]+|lr]], #16711680
; PRE6-NEXT:  and [[R2:r[0-9]+|lr]], [[R1]], [[R0:r[0-9]]], lsl #8
; PRE6-NEXT:  orr [[R3:r[0-9]+|lr]], [[R2]], [[R0]], lsl #24
; PRE6-NEXT:  lsr [[R4:r[0-9]+|lr]], [[R3]], #16

; REV-LABEL: bswap16
;; FIXME: It would ben nice if DAG legalization was taught to not
;; promote the incoming reg to i32 in this case, so that the
;; combiner could canonicalize this to (rotr (bswap x), 16), which
;; would then get matched as REV16.
; REV:      rev [[R0:r[0-9]+|lr]], {{r[0-9]+}}
; REV-NEXT: lsr {{r[0-9]+}}, [[R0]], #16
}

define i32 @bswap32(i32 %x) #0 {
  %1 = tail call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %1
; PRE6-LABEL: bswap32
; PRE6-DAG:   mov [[R0:r[0-9]+|lr]], #65280
; PRE6-NOT:   DAG-BREAK!
; PRE6-DAG:   and [[R0]], [[R0]], [[R1:r[0-9]+|lr]], lsr #8
; PRE6-DAG:   orr [[R0]], [[R0]], [[R1]], lsr #24
; PRE6-DAG:   mov [[R2:r[0-9]+|lr]], #16711680
; PRE6-DAG:   and [[R2]], [[R2]], [[R1]], lsl #8
; PRE6-DAG:   orr [[R1]], [[R2]], [[R1]], lsl #24
; PRE6-NOT:   DAG-BREAK!
; PRE6-DAG:   orr [[R1]], [[R1]], [[R0]]

; REV-LABEL: bswap32
; REV:       rev {{r[0-9]+|lr}}, {{r[0-9]+|lr}}
; REV-NOT:   rev
}

define i48 @bswap48(i48 %x) #0 {
  %1 = tail call i48 @llvm.bswap.i48(i48 %x)
  ret i48 %1
; PRE6-LABEL: bswap48
; PRE6-DAG:   mov [[R0:r[0-9]+|lr]], #65280
; PRE6-DAG:   mov [[R1:r[0-9]+|lr]], #16711680
; PRE6-NOT:   DAG-BREAK!
; PRE6-DAG:   and [[R0]], [[R0]], [[R2:r[0-9]+|lr]], lsr #8
; PRE6-DAG:   and [[R3:r[0-9]+|lr]], [[R1]], [[R2]], lsl #8
; PRE6-DAG:   orr [[R0]], [[R0]], [[R2]], lsr #24
; PRE6-DAG:   orr [[R2]], [[R3]], [[R2]], lsl #24
; PRE6-DAG:   orr [[R0]], [[R2]], [[R0]]
; PRE6-NOT:   DAG-BREAK!
; PRE6-DAG:   and [[R2]], [[R1]], [[R4:r[0-9]+|lr]], lsl #8
; PRE6-DAG:   orr [[R2]], [[R2]], [[R4]], lsl #24
; PRE6-DAG:   lsr [[R4]], [[R0]], #16
; PRE6-DAG:   lsr [[R2]], [[R2]], #16
; PRE6-NOT:   DAG-BREAK!
; PRE6-DAG:   orr [[R2]], [[R2]], [[R0]], lsl #16

; PRE6BE-LABEL: bswap48
; PRE6BE-DAG: mov [[R0:r[0-9]+|lr]], #65280
; PRE6BE-DAG: mov [[R1:r[0-9]+|lr]], #16711680
; PRE6-NOT:   DAG-BREAK!
; PRE6BE-DAG: and [[R0]], [[R0]], [[R2:r[0-9]+|lr]], lsr #8
; PRE6BE-DAG: and [[R3:r[0-9]+|lr]], [[R1]], [[R2:r[0-9]+|lr]], lsl #8
; PRE6BE-DAG: orr [[R0]], [[R0]], [[R2]], lsr #24
; PRE6BE-DAG: orr [[R2]], [[R3]], [[R2]], lsl #24
; PRE6BE-DAG: orr [[R0]], [[R2]], [[R0]]
; PRE6-NOT:   DAG-BREAK!
; PRE6BE-DAG: and [[R2]], [[R1]], [[R4:r[0-9]+|lr]], lsl #8
; PRE6BE-DAG: orr [[R4]], [[R2]], [[R4]], lsl #24
; PRE6BE-DAG: lsr [[R4]], [[R4]], #16
; PRE6BE-DAG: lsr [[R4]], [[R0]], #16
; PRE6-NOT:   DAG-BREAK!
; PRE6BE-DAG: orr [[R2]], [[R4]], [[R0]], lsl #16

; REV-LABEL: bswap48
; REV-DAG:   rev [[R0:r[0-9]+]], [[R1:r[0-9]+]]
; REV-DAG:   rev [[R1]], [[R2:r[0-9]+]]
; REV-DAG:   lsr [[R1]], [[R1]], #16
; REV-DAG:   lsr [[R2]], [[R0]], #16
; REV-DAG:   orr [[R1]], [[R1]], [[R0]], lsl #16

; REVBE-LABEL: bswap48
; Until PR24879 is fixed, I can't match [[R0:r[0-9]+|lr]] in a
; backreference. Having to stick to just r[0-9]+ for now, which
; is *very* likely to be the register selection :-).
; REVBE-DAG: rev [[R0:r[0-9]+]], [[R0]]
; Need to break DAG block here to stop R1 or R2 clobbering the
; self rev above.
; REVBE-NEXT: rev [[R1:r[0-9]+|lr]], [[R2:r[0-9]+|lr]]
; REVBE-DAG: lsr [[R0]], [[R0]], #16
; REVBE-DAG: lsr [[R0]], [[R1]], #16
; REVBE-DAG: orr [[R2]], [[R0]], [[R1]], lsl #16
}

define i64 @bswap64(i64 %x) #0 {
  %1 = tail call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %1
; PRE6-LABEL: bswap64
; PRE6-DAG: mov [[R0:r[0-9]+|lr]], #65280
; PRE6-DAG: mov [[R1:r[0-9]+|lr]], #16711680
; PRE6-NOT:   DAG-BREAK!
; PRE6-DAG: and [[R2:r[0-9]+|lr]], [[R0]], [[R3:r[0-9]+|lr]], lsr #8
; PRE6-DAG: and [[R4:r[0-9]+|lr]], [[R1]], [[R3]], lsl #8
; PRE6-DAG: orr [[R2]], [[R2]], [[R3]], lsr #24
; PRE6:     orr [[R3]], [[R4]], [[R3]], lsl #24
; PRE6-NOT: DAG-BREAK!
; PRE6-DAG: and [[R4]], [[R1]], [[R5:r[0-9]+|lr]], lsl #8
; PRE6-DAG: orr [[R2]], [[R3]], [[R2]]
; PRE6-DAG: and [[R3]], [[R0]], [[R5]], lsr #8
; PRE6-DAG: orr [[R3]], [[R3]], [[R5]], lsr #24
; PRE6-DAG: orr [[R5]], [[R4]], [[R5]], lsl #24
; PRE6-DAG: orr [[R3]], [[R5]], [[R3]]

; REV-LABEL: bswap64
; Just check that the two 32-bit words are reversed, not bothered
; so much about regiter selection here.
; REV:     rev
; REV:     rev
; REV-NOT: rev
}

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare i48 @llvm.bswap.i48(i48)
declare i64 @llvm.bswap.i64(i64)
