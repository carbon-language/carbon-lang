; This code snippet is derived from the following "whetstone" style code:
; int whet(int j) {
;   if (j == 1) j = 2;
;   else        j = 7;
;   if (j > 2)  j = 0;
;   else        j = 3;
;   if (j < 1)  j = 10;
;   else        j = 0;
;   return j;
;}
;
; This should eliminate all BB's except BB0, BB9, BB10
;
; RUN: as < %s | opt -cee -instcombine -simplifycfg | dis | not grep 'bb[2-8]'

implementation   ; Functions:

int %whet(int %j) {
bb0:		; No predecessors!
	%cond220 = setne int %j, 1		; <bool> [#uses=1]
	br bool %cond220, label %bb3, label %bb4

bb3:		; preds = %bb0
	br label %bb4

bb4:		; preds = %bb3, %bb0
	%reg111 = phi int [ 7, %bb3 ], [ 2, %bb0 ]		; <int> [#uses=1]
	%cond222 = setle int %reg111, 2		; <bool> [#uses=1]
	br bool %cond222, label %bb6, label %bb7

bb6:		; preds = %bb4
	br label %bb7

bb7:		; preds = %bb6, %bb4
	%reg114 = phi int [ 3, %bb6 ], [ 0, %bb4 ]		; <int> [#uses=1]
	%cond225 = setgt int %reg114, 0		; <bool> [#uses=1]
	br bool %cond225, label %bb9, label %bb10

bb9:		; preds = %bb7
	br label %bb10

bb10:		; preds = %bb9, %bb7
	%reg117 = phi int [ 0, %bb9 ], [ 10, %bb7 ]		; <int> [#uses=1]
	ret int %reg117
}
