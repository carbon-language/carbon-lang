; RUN: llc < %s -mtriple=i386-apple-macosx -mcpu=penryn | FileCheck %s
; PR3253

; The register+memory form of the BT instruction should be usable on
; pentium4, however it is currently disabled due to the register+memory
; form having different semantics than the register+register form.

; Test these patterns:
;    (X & (1 << N))  != 0  -->  BT(X, N).
;    ((X >>u N) & 1) != 0  -->  BT(X, N).
; as well as several variations:
;    - The second form can use an arithmetic shift.
;    - Either form can use == instead of !=.
;    - Either form can compare with an operand of the &
;      instead of with 0.
;    - The comparison can be commuted (only cases where neither
;      operand is constant are included).
;    - The and can be commuted.

define void @test2(i32 %x, i32 %n) nounwind {
entry:
; CHECK: test2
; CHECK: btl %eax, %ecx
; CHECK: jb
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test2b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: test2b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @atest2(i32 %x, i32 %n) nounwind {
entry:
; CHECK: atest2
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @atest2b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: atest2b
; CHECK: btl %e{{..}}, %e{{..}}
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test3(i32 %x, i32 %n) nounwind {
entry:
; CHECK: test3
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test3b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: test3b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @testne2(i32 %x, i32 %n) nounwind {
entry:
; CHECK: testne2
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @testne2b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: testne2b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @atestne2(i32 %x, i32 %n) nounwind {
entry:
; CHECK: atestne2
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @atestne2b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: atestne2b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @testne3(i32 %x, i32 %n) nounwind {
entry:
; CHECK: testne3
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @testne3b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: testne3b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query2(i32 %x, i32 %n) nounwind {
entry:
; CHECK: query2
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query2b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: query2b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @aquery2(i32 %x, i32 %n) nounwind {
entry:
; CHECK: aquery2
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @aquery2b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: aquery2b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query3(i32 %x, i32 %n) nounwind {
entry:
; CHECK: query3
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, %tmp29		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query3b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: query3b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp eq i32 %tmp3, %tmp29		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query3x(i32 %x, i32 %n) nounwind {
entry:
; CHECK: query3x
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp29, %tmp3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query3bx(i32 %x, i32 %n) nounwind {
entry:
; CHECK: query3bx
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jae
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp eq i32 %tmp29, %tmp3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne2(i32 %x, i32 %n) nounwind {
entry:
; CHECK: queryne2
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne2b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: queryne2b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @aqueryne2(i32 %x, i32 %n) nounwind {
entry:
; CHECK: aqueryne2
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @aqueryne2b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: aqueryne2b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne3(i32 %x, i32 %n) nounwind {
entry:
; CHECK: queryne3
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, %tmp29		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne3b(i32 %x, i32 %n) nounwind {
entry:
; CHECK: queryne3b
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp ne i32 %tmp3, %tmp29		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne3x(i32 %x, i32 %n) nounwind {
entry:
; CHECK: queryne3x
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp29, %tmp3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne3bx(i32 %x, i32 %n) nounwind {
entry:
; CHECK: queryne3bx
; CHECK: btl %e{{..}}, %e{{..}}
; CHECK: jb
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp ne i32 %tmp29, %tmp3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare void @foo()

define zeroext i1 @invert(i32 %flags, i32 %flag) nounwind {
; CHECK: btl
entry:
  %neg = xor i32 %flags, -1
  %shl = shl i32 1, %flag
  %and = and i32 %shl, %neg
  %tobool = icmp ne i32 %and, 0
  ret i1 %tobool
}
