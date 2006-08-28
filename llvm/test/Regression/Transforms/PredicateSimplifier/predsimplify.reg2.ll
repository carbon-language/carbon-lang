; RUN: llvm-as < %s | opt -predsimplify -verify

; ModuleID = 'bugpoint-reduced-simplified.bc'
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
deplibs = [ "c", "crtend" ]
	%struct.anon = type { %struct.set_family*, %struct.set_family*, %struct.set_family*, sbyte*, int, uint*, %struct.pair_struct*, sbyte**, %struct.symbolic_t*, %struct.symbolic_t* }
	%struct.pair_struct = type { int, int*, int* }
	%struct.set_family = type { int, int, int, int, int, uint*, %struct.set_family* }
	%struct.symbolic_label_t = type { sbyte*, %struct.symbolic_label_t* }
	%struct.symbolic_list_t = type { int, int, %struct.symbolic_list_t* }
	%struct.symbolic_t = type { %struct.symbolic_list_t*, int, %struct.symbolic_label_t*, int, %struct.symbolic_t* }

implementation   ; Functions:

void %find_pairing_cost(int %strategy) {
entry:
	br bool false, label %cond_true299, label %bb314

bb94:		; preds = %cond_true299
	switch int %strategy, label %bb246 [
		 int 0, label %bb196
		 int 1, label %bb159
	]

cond_next113:		; preds = %cond_true299
	switch int %strategy, label %bb246 [
		 int 0, label %bb196
		 int 1, label %bb159
	]

bb159:		; preds = %cond_next113, %bb94
	ret void

bb196:		; preds = %cond_next113, %bb94
	%Rsave.0.3 = phi %struct.set_family* [ null, %bb94 ], [ null, %cond_next113 ]		; <%struct.set_family*> [#uses=0]
	ret void

bb246:		; preds = %cond_next113, %bb94
	br label %bb314

cond_true299:		; preds = %entry
	%tmp55 = setgt int %strategy, 0		; <bool> [#uses=1]
	br bool %tmp55, label %bb94, label %cond_next113

bb314:		; preds = %bb246, %entry
	ret void
}
