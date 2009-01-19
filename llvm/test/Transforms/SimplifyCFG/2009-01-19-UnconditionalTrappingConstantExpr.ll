; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep {br i1 } | count 2
; PR3354
; Do not merge bb1 into the entry block, it might trap.

@G = extern_weak global i32

define i32 @test(i32 %tmp21, i32 %tmp24) {
	%tmp25 = icmp sle i32 %tmp21, %tmp24		
	br i1 %tmp25, label %bb2, label %bb1	
					
bb1:		; preds = %bb	
	%tmp26 = icmp sgt i32 sdiv (i32 -32768, i32 ptrtoint (i32* @G to i32)), 0
	br i1 %tmp26, label %bb6, label %bb2		
bb2:
	ret i32 42

bb6:
	unwind
}

