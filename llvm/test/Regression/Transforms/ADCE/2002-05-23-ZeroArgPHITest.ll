; This testcase contains a entire loop that should be removed.  The only thing
; left is the store instruction in BB0.  The problem this testcase was running
; into was that when the reg109 PHI was getting zero predecessors, it was 
; removed even though there were uses still around.  Now the uses are filled
; in with a dummy value before the PHI is deleted.
;
; RUN: as < %s | opt -adce
	
%node_t = type { double*, %node_t*, %node_t**, double**, double*, int, int }

implementation   ; Functions:

void %localize_local(%node_t* %nodelist) {
bb0:					;[#uses=0]
	%nodelist = alloca %node_t*		; <%node_t**> [#uses=2]
	store %node_t* %nodelist, %node_t** %nodelist
	br label %bb1

bb1:					;[#uses=2]
	%reg107 = load %node_t** %nodelist		; <%node_t*> [#uses=2]
	%cond211 = seteq %node_t* %reg107, null		; <bool> [#uses=1]
	br bool %cond211, label %bb3, label %bb2

bb2:					;[#uses=3]
	%reg109 = phi %node_t* [ %reg110, %bb2 ], [ %reg107, %bb1 ]		; <%node_t*> [#uses=1]
	%reg212 = getelementptr %node_t* %reg109, long 0, ubyte 1		; <%node_t**> [#uses=1]
	%reg110 = load %node_t** %reg212		; <%node_t*> [#uses=2]
	%cond213 = setne %node_t* %reg110, null		; <bool> [#uses=1]
	br bool %cond213, label %bb2, label %bb3

bb3:					;[#uses=2]
	ret void
}
