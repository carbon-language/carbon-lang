; This test shows SCCP "proving" that the loop (from bb6 to 14) loops infinitely
; this is in fact NOT the case, so the return should still be alive in the code
; after sccp and CFG simplification have been performed.
;
; RUN: llvm-as < %s | opt -sccp -simplifycfg | llvm-dis | grep ret


void "old_main"() {
bb3:					;[#uses=1]
	br label %bb6

bb6:					;[#uses=3]
	%reg403 = phi int [ %reg155, %bb14 ], [ 0, %bb3 ]		; <int> [#uses=2]
	%reg155 = add int %reg403, 1		; <int> [#uses=3]

	br label %bb11

bb11:
        %reg407 = phi int [ %reg408, %bb11 ], [ 0, %bb6 ]              ; <int> [#uses=2]
        %reg408 = add int %reg407, 1            ; <int> [#uses=2]
        %cond550 = setle int %reg407, 1         ; <bool> [#uses=1]
        br bool %cond550, label %bb11, label %bb12

bb12:					;[#uses=2]
	br label %bb13

bb13:					;[#uses=3]
	%reg409 = phi int [ %reg410, %bb13 ], [ 0, %bb12 ]		; <int> [#uses=1]
	%reg410 = add int %reg409, 1		; <int> [#uses=2]
	%cond552 = setle int %reg410, 2		; <bool> [#uses=1]
	br bool %cond552, label %bb13, label %bb14

bb14:					;[#uses=2]
	%cond553 = setle int %reg155, 31		; <bool> [#uses=1]
	br bool %cond553, label %bb6, label %bb15

bb15:					;[#uses=1]
	ret void
}
