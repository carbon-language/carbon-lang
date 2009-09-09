; RUN: llc < %s -march=bfin -verify-machineinstrs

; This somewhat contrived function heavily exercises register classes
; It can trick -join-cross-class-copies into making illegal joins

define void @f(i16** nocapture %p) nounwind readonly {
entry:
	%tmp1 = load i16** %p		; <i16*> [#uses=1]
	%tmp2 = load i16* %tmp1		; <i16> [#uses=1]
	%ptr = getelementptr i16* %tmp1, i16 %tmp2
    store i16 %tmp2, i16* %ptr
    ret void
}
