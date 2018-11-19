; RUN: opt -S -licm < %s | FileCheck %s
; RUN: opt -passes='require<opt-remark-emit>,loop(licm)' -S < %s | FileCheck %s

; CHECK-LABEL: @triangle_phi
define void @triangle_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp sgt i32 %x, 0
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: %add = add i32 %x, 1
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: phi i32 [ %add, %[[IF_LICM]] ], [ %x, %entry ]
; CHECK: store i32 %phi, i32* %p
; CHECK: %cmp2 = icmp ne i32 %phi, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %add = add i32 %x, 1
  br label %then

then:
  %phi = phi i32 [ %add, %if ], [ %x, %loop ]
  store i32 %phi, i32* %p
  %cmp2 = icmp ne i32 %phi, 0
  br i1 %cmp2, label %loop, label %end

end:
  ret void
}

; CHECK-LABEL: @diamond_phi
define void @diamond_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp sgt i32 %x, 0
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: %add = add i32 %x, 1
; CHECK: br label %[[THEN_LICM:.*]]

; CHECK: [[ELSE_LICM]]:
; CHECK: %sub = sub i32 %x, 1
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]
; CHECK: %phi = phi i32 [ %add, %[[IF_LICM]] ], [ %sub, %[[ELSE_LICM]] ]
; CHECK: store i32 %phi, i32* %p
; CHECK: %cmp2 = icmp ne i32 %phi, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  br label %then

else:
  %sub = sub i32 %x, 1
  br label %then

then:
  %phi = phi i32 [ %add, %if ], [ %sub, %else ]
  store i32 %phi, i32* %p
  %cmp2 = icmp ne i32 %phi, 0
  br i1 %cmp2, label %loop, label %end

end:
  ret void
}

; TODO: This is currently too complicated for us to be able to hoist the phi.
; CHECK-LABEL: @three_way_phi
define void @three_way_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp2 = icmp sgt i32 %add, 0
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM:.*]]

; CHECK: [[THEN_LICM]]:
; CHECK: %sub = sub i32 %x, 1
; CHECK: br label %loop

entry:
  br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 %add, 0
  br i1 %cmp2, label %if.if, label %then

if.if:
  %sub = sub i32 %x, 1
  br label %then

then:
  %phi = phi i32 [ 0, %loop ], [ %add, %if ], [ %sub, %if.if ]
  store i32 %phi, i32* %p
  %cmp3 = icmp ne i32 %phi, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; TODO: This is currently too complicated for us to be able to hoist the phi.
; CHECK-LABEL: @tree_phi
define void @tree_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp2 = icmp sgt i32 %add, 0
; CHECK-DAG: %sub = sub i32 %x, 1
; CHECK: br label %loop

entry:
  br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 %add, 0
  br i1 %cmp2, label %if.if, label %if.else

if.if:
  br label %then

if.else:
  br label %then

else:
  %sub = sub i32 %x, 1
  br label %then

then:
  %phi = phi i32 [ %add, %if.if ], [ 0, %if.else ], [ %sub, %else ]
  store i32 %phi, i32* %p
  %cmp3 = icmp ne i32 %phi, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; TODO: We can hoist the first phi, but not the second.
; CHECK-LABEL: @phi_phi
define void @phi_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp2 = icmp sgt i32 %add, 0
; CHECK-DAG: %sub = sub i32 %x, 1
; CHECK: br i1 %cmp2, label %[[IF_IF_LICM:.*]], label %[[IF_ELSE_LICM:.*]]

; CHECK: [[IF_IF_LICM]]:
; CHECK: br label %[[IF_THEN_LICM:.*]]

; CHECK: [[IF_ELSE_LICM]]:
; CHECK: br label %[[IF_THEN_LICM]]

; CHECK: [[IF_THEN_LICM]]:
; CHECK: %phi1 = phi i32 [ %add, %[[IF_IF_LICM]] ], [ 0, %[[IF_ELSE_LICM]] ]
; CHECK: br label %loop

entry:
  br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 %add, 0
  br i1 %cmp2, label %if.if, label %if.else

if.if:
  br label %if.then

if.else:
  br label %if.then

if.then:
  %phi1 = phi i32 [ %add, %if.if ], [ 0, %if.else ]
  br label %then

else:
  %sub = sub i32 %x, 1
  br label %then

then:
  %phi2 = phi i32 [ %phi1, %if.then ], [ %sub, %else ]
  store i32 %phi2, i32* %p
  %cmp3 = icmp ne i32 %phi2, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; Check that we correctly duplicate empty control flow.
; CHECK-LABEL: @empty_triangle_phi
define i8 @empty_triangle_phi(i32 %x, i32 %y) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp eq i32 %x, 0
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %entry ]
; CHECK: %cmp2 = icmp eq i32 %y, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp eq i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  br label %then

then:
  %phi = phi i8 [ 0, %if ], [ 1, %loop ]
  %cmp2 = icmp eq i32 %y, 0
  br i1 %cmp2, label %end, label %loop

end:
  ret i8 %phi
}

; CHECK-LABEL: @empty_diamond_phi
define i8 @empty_diamond_phi(i32 %x, i32 %y) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp eq i32 %x, 0
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM:.*]]

; CHECK: [[ELSE_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %[[ELSE_LICM]] ]
; CHECK: %cmp2 = icmp eq i32 %y, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp eq i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  br label %then

else:
  br label %then

then:
  %phi = phi i8 [ 0, %if ], [ 1, %else ]
  %cmp2 = icmp eq i32 %y, 0
  br i1 %cmp2, label %end, label %loop

end:
  ret i8 %phi
}

; Check that we correctly handle the case that the first thing we try to hoist is a phi.
; CHECK-LABEL: @empty_triangle_phi_first
define i8 @empty_triangle_phi_first(i32 %x, i1 %cond) {
; CHECK-LABEL: entry:
; CHECK: br i1 %cond, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %entry ]
; CHECK: %cmp = icmp eq i32 %x, 0
; CHECK: br label %loop

loop:
  br i1 %cond, label %if, label %then

if:
  br label %then

then:
  %phi = phi i8 [ 0, %if ], [ 1, %loop ]
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %end, label %loop

end:
  ret i8 %phi
}

; CHECK-LABEL: @empty_diamond_phi
define i8 @empty_diamond_phi_first(i32 %x, i1 %cond) {
; CHECK-LABEL: entry:
; CHECK: br i1 %cond, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM:.*]]

; CHECK: [[ELSE_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %[[ELSE_LICM]] ]
; CHECK: %cmp = icmp eq i32 %x, 0
; CHECK: br label %loop

loop:
  br i1 %cond, label %if, label %else

if:
  br label %then

else:
  br label %then

then:
  %phi = phi i8 [ 0, %if ], [ 1, %else ]
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %end, label %loop

end:
  ret i8 %phi
}

; CHECK-LABEL: @empty_triangle_phi_first
define i8 @empty_triangle_phi_first_empty_loop_head(i32 %x, i1 %cond) {
; CHECK-LABEL: entry:
; CHECK: br i1 %cond, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %entry ]
; CHECK: %cmp = icmp eq i32 %x, 0
; CHECK: br label %loop

loop:
  br label %test

test:
  br i1 %cond, label %if, label %then

if:
  br label %then

then:
  %phi = phi i8 [ 0, %if ], [ 1, %test ]
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %end, label %loop

end:
  ret i8 %phi
}

; CHECK-LABEL: @empty_diamond_phi_first_empty_loop_head
define i8 @empty_diamond_phi_first_empty_loop_head(i32 %x, i1 %cond) {
; CHECK-LABEL: entry:
; CHECK: br i1 %cond, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM:.*]]

; CHECK: [[ELSE_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %[[ELSE_LICM]] ]
; CHECK: %cmp = icmp eq i32 %x, 0
; CHECK: br label %loop

loop:
  br label %test

test:
  br i1 %cond, label %if, label %else

if:
  br label %then

else:
  br label %then

then:
  %phi = phi i8 [ 0, %if ], [ 1, %else ]
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %end, label %loop

end:
  ret i8 %phi
}

; The phi is on one branch of a diamond while simultaneously at the end of a
; triangle. Check that we duplicate the triangle and not the diamond.
; CHECK-LABEL: @triangle_diamond
define void @triangle_diamond(i32* %ptr, i32 %x, i32 %y) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 %x, 0
; CHECK-DAG: %cmp2 = icmp ne i32 %y, 0
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i32 [ 0, %[[IF_LICM]] ], [ 127, %entry ]

loop:
  %cmp1 = icmp ne i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %cmp2 = icmp ne i32 %y, 0
  br i1 %cmp2, label %if.then, label %then

then:
  %phi = phi i32 [ 0, %if ], [ 127, %loop ]
  store i32 %phi, i32* %ptr
  br label %end

if.then:
  br label %end

end:
  br label %loop
}

; As the previous, but the end of the diamond is the head of the loop.
; CHECK-LABEL: @triangle_diamond_backedge
define void @triangle_diamond_backedge(i32* %ptr, i32 %x, i32 %y) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 %x, 0
; CHECK-DAG: %cmp2 = icmp ne i32 %y, 0
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i32 [ 0, %[[IF_LICM]] ], [ 127, %entry ]

loop:
  %cmp1 = icmp ne i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %cmp2 = icmp ne i32 %y, 0
  br i1 %cmp2, label %backedge, label %then

then:
  %phi = phi i32 [ 0, %if ], [ 127, %loop ]
  store i32 %phi, i32* %ptr
  br label %loop

backedge:
  br label %loop
}

; TODO: The inner diamonds can be hoisted, but not currently the outer diamond
; CHECK-LABEL: @diamonds_inside_diamond
define void @diamonds_inside_diamond(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %cmp3 = icmp slt i32 %x, -10
; CHECK: br i1 %cmp3, label %[[ELSE_IF_LICM:.*]], label %[[ELSE_ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK: [[ELSE_IF_LICM]]:
; CHECK: br label %[[ELSE_THEN_LICM:.*]]

; CHECK: [[ELSE_ELSE_LICM]]:
; CHECK: br label %[[ELSE_THEN_LICM]]

; CHECK: [[ELSE_THEN_LICM]]:
; CHECK: %phi2 = phi i32 [ 2, %[[ELSE_IF_LICM]] ], [ 3, %[[ELSE_ELSE_LICM]] ]
; CHECK: %cmp2 = icmp sgt i32 %x, 10
; CHECK: br i1 %cmp2, label %[[IF_IF_LICM:.*]], label %[[IF_ELSE_LICM:.*]]

; CHECK: [[IF_IF_LICM]]:
; CHECK: br label %[[IF_THEN_LICM:.*]]

; CHECK: [[IF_ELSE_LICM]]:
; CHECK: br label %[[IF_THEN_LICM]]

; CHECK: [[IF_THEN_LICM]]:
; CHECK: %phi1 = phi i32 [ 0, %[[IF_IF_LICM]] ], [ 1, %[[IF_ELSE_LICM]] ]
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %cmp2 = icmp sgt i32 %x, 10
  br i1 %cmp2, label %if.if, label %if.else

if.if:
  br label %if.then

if.else:
  br label %if.then

if.then:
  %phi1 = phi i32 [ 0, %if.if ], [ 1, %if.else ]
  br label %then

else:
  %cmp3 = icmp slt i32 %x, -10
  br i1 %cmp3, label %else.if, label %else.else

else.if:
  br label %else.then

else.else:
  br label %else.then

else.then:
  %phi2 = phi i32 [ 2, %else.if ], [ 3, %else.else ]
  br label %then

then:
  %phi3 = phi i32 [ %phi1, %if.then ], [ %phi2, %else.then ]
  store i32 %phi3, i32* %p
  %cmp4 = icmp ne i32 %phi3, 0
  br i1 %cmp4, label %loop, label %end

end:
  ret void
}

; We can hoist blocks that contain an edge that exits the loop by ignoring that
; edge in the hoisted block.
; CHECK-LABEL: @triangle_phi_loopexit
define void @triangle_phi_loopexit(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %cmp2 = icmp sgt i32 10, %add
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi = phi i32 [ %add, %[[IF_LICM]] ], [ %x, %entry ]

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 10, %add
  br i1 %cmp2, label %then, label %end

then:
  %phi = phi i32 [ %add, %if ], [ %x, %loop ]
  store i32 %phi, i32* %p
  %cmp3 = icmp ne i32 %phi, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; CHECK-LABEL: @diamond_phi_oneloopexit
define void @diamond_phi_oneloopexit(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %cmp2 = icmp sgt i32 10, %add
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM:.*]]

; CHECK: [[ELSE_LICM]]:
; CHECK: %sub = sub i32 %x, 1
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]
; CHECK: %phi = phi i32 [ %add, %[[IF_LICM]] ], [ %sub, %[[ELSE_LICM]] ]
; CHECK: %cmp3 = icmp ne i32 %phi, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 10, %add
  br i1 %cmp2, label %then, label %end

else:
  %sub = sub i32 %x, 1
  br label %then

then:
  %phi = phi i32 [ %add, %if ], [ %sub, %else ]
  store i32 %phi, i32* %p
  %cmp3 = icmp ne i32 %phi, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; CHECK-LABEL: @diamond_phi_twoloopexit
define void @diamond_phi_twoloopexit(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %sub = sub i32 %x, 1
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %cmp2 = icmp sgt i32 10, %add
; CHECK-DAG: %cmp3 = icmp sgt i32 10, %sub
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM:.*]]

; CHECK: [[ELSE_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]
; CHECK: %phi = phi i32 [ %add, %[[IF_LICM]] ], [ %sub, %[[ELSE_LICM]] ]
; CHECK: %cmp4 = icmp ne i32 %phi, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 10, %add
  br i1 %cmp2, label %then, label %end

else:
  %sub = sub i32 %x, 1
  %cmp3 = icmp sgt i32 10, %sub
  br i1 %cmp3, label %then, label %end

then:
  %phi = phi i32 [ %add, %if ], [ %sub, %else ]
  store i32 %phi, i32* %p
  %cmp4 = icmp ne i32 %phi, 0
  br i1 %cmp4, label %loop, label %end

end:
  ret void
}

; The store cannot be hoisted, so add and shr cannot be hoisted into a
; conditional block.
; CHECK-LABEL: @conditional_use
define void @conditional_use(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cond = icmp ugt i32 %x, 0
; CHECK-DAG: %add = add i32 %x, 5
; CHECK-DAG: %shr = ashr i32 %add, 1
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %cond = icmp ugt i32 %x, 0
  br i1 %cond, label %if, label %else

; CHECK-LABEL: if:
; CHECK: store i32 %shr, i32* %p, align 4
if:
  %add = add i32 %x, 5
  %shr = ashr i32 %add, 1
  store i32 %shr, i32* %p, align 4
  br label %then

else:
  br label %then

then:
  br label %loop
}

; A diamond with two triangles on the left and one on the right. This test is
; to check that we have a unique loop preheader when we hoist the store (and so
; don't fail an assertion).
; CHECK-LABEL: @triangles_in_diamond
define void @triangles_in_diamond(i32* %ptr) {
; CHECK-LABEL: entry:
; CHECK: store i32 0, i32* %ptr, align 4
; CHECK: br label %loop
entry:
  br label %loop

loop:
  br i1 undef, label %left_triangle_1, label %right_triangle

left_triangle_1:
  br i1 undef, label %left_triangle_1_if, label %left_triangle_2

left_triangle_1_if:
  br label %left_triangle_2

left_triangle_2:
  br i1 undef, label %left_triangle_2_if, label %left_triangle_2_then

left_triangle_2_if:
  br label %left_triangle_2_then

left_triangle_2_then:
  br label %loop.end

right_triangle:
  br i1 undef, label %right_triangle.if, label %right_triangle.then

right_triangle.if:
  br label %right_triangle.then

right_triangle.then:
  br label %loop.end

loop.end:
  store i32 0, i32* %ptr, align 4
  br label %loop
}

; %cmp dominates its used after being hoisted, but not after %brmerge is rehoisted
; CHECK-LABEL: @rehoist
define void @rehoist(i8* %this, i32 %x) {
; CHECK-LABEL: entry:
; CHECK-DAG: %sub = add nsw i32 %x, -1
; CHECK-DAG: %fptr = bitcast i8* %this to void (i8*)*
; CHECK-DAG: %cmp = icmp eq i32 0, %sub
; CHECK-DAG: %brmerge = or i1 %cmp, true
entry:
  %sub = add nsw i32 %x, -1
  br label %loop

loop:
  br i1 undef, label %if1, label %else1

if1:
  %fptr = bitcast i8* %this to void (i8*)*
  call void %fptr(i8* %this)
  br label %then1

else1:
  br label %then1

then1:
  %cmp = icmp eq i32 0, %sub
  br i1 %cmp, label %end, label %else2

else2:
  %brmerge = or i1 %cmp, true
  br i1 %brmerge, label %if3, label %end

if3:
  br label %end

end:
  br label %loop
}

; A test case that uses empty blocks in a way that can cause control flow
; hoisting to get confused.
; CHECK-LABEL: @empty_blocks_multiple_conditional_branches
define void @empty_blocks_multiple_conditional_branches(float %arg, float* %ptr) {
; CHECK-LABEL: entry
; CHECK-DAG: %div1 = fmul float %arg, 4.000000e+00
; CHECK-DAG: %div2 = fmul float %arg, 2.000000e+00
entry:
  br label %loop

; The exact path to the phi isn't checked here, because it depends on whether
; cond2 or cond3 is hoisted first
; CHECK: %phi = phi float [ 0.000000e+00, %{{.*}} ], [ %div1, %{{.*}} ]
; CHECK: br label %loop

loop:
  br i1 undef, label %backedge2, label %cond1

cond1:
  br i1 undef, label %cond1.if, label %cond1.else

cond1.else:
  br label %cond3

cond1.if:
  br label %cond1.if.next

cond1.if.next:
  br label %cond2

cond2:
  %div1 = fmul float %arg, 4.000000e+00
  br i1 undef, label %cond2.if, label %cond2.then

cond2.if:
  br label %cond2.then

cond2.then:
  %phi = phi float [ 0.000000e+00, %cond2 ], [ %div1, %cond2.if ]
  store float %phi, float* %ptr
  br label %backedge2

cond3:
  br i1 undef, label %cond3.then, label %cond3.if

cond3.if:
  %div2 = fmul float %arg, 2.000000e+00
  store float %div2, float* %ptr
  br label %cond3.then

cond3.then:
  br label %loop

backedge2:
  br label %loop
}

; We can't do much here, so mainly just check that we don't crash.
; CHECK-LABEL: @many_path_phi
define void @many_path_phi(i32* %ptr1, i32* %ptr2) {
; CHECK-LABEL: entry:
; CHECK-DAG: %gep3 = getelementptr inbounds i32, i32* %ptr2, i32 2
; CHECK-DAG: %gep2 = getelementptr inbounds i32, i32* %ptr2, i32 2
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %phi1 = phi i32 [ 0, %entry ], [ %phi2, %end ]
  %cmp1 = icmp ugt i32 %phi1, 3
  br i1 %cmp1, label %cond2, label %cond1

cond1:
  br i1 undef, label %end, label %cond1.else

cond1.else:
  %gep2 = getelementptr inbounds i32, i32* %ptr2, i32 2
  %val2 = load i32, i32* %gep2, align 4
  %cmp2 = icmp eq i32 %val2, 13
  br i1 %cmp2, label %cond1.end, label %end

cond1.end:
  br label %end

cond2:
  br i1 undef, label %end, label %cond2.else

cond2.else:
  %gep3 = getelementptr inbounds i32, i32* %ptr2, i32 2
  %val3 = load i32, i32* %gep3, align 4
  %cmp3 = icmp eq i32 %val3, 13
  br i1 %cmp3, label %cond2.end, label %end

cond2.end:
  br label %end

end:
  %phi2 = phi i32 [ 1, %cond1 ], [ 2, %cond1.else ], [ 3, %cond1.end ], [ 4, %cond2 ], [ 5, %cond2.else ], [ 6, %cond2.end ]
  br label %loop
}

; Check that we correctly handle the hoisting of %gep when theres a critical
; edge that branches to the preheader.
; CHECK-LABEL: @crit_edge
define void @crit_edge(i32* %ptr, i32 %idx, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: entry:
; CHECK: %gep = getelementptr inbounds i32, i32* %ptr, i32 %idx
; CHECK: br label %preheader
entry:
  br label %preheader

preheader:
  br label %loop

loop:
  br i1 %cond1, label %then, label %if

if:
  %gep = getelementptr inbounds i32, i32* %ptr, i32 %idx
  %val = load i32, i32* %gep
  br label %then

then:
  %phi = phi i32 [ %val, %if ], [ 0, %loop ]
  store i32 %phi, i32* %ptr
  br i1 %cond2, label %loop, label %crit_edge

crit_edge:
  br label %preheader
}

; Check that the conditional sub is correctly hoisted from the inner loop to the
; preheader of the outer loop.
; CHECK-LABEL: @hoist_from_innermost_loop
define void @hoist_from_innermost_loop(i32 %nx, i32* %ptr) {
; CHECK-LABEL: entry:
; CHECK-DAG: %sub = sub nsw i32 0, %nx
; CHECK: br label %outer_loop
entry:
  br label %outer_loop

outer_loop:
  br label %middle_loop

middle_loop:
  br label %inner_loop

inner_loop:
  br i1 undef, label %inner_loop_end, label %if

if:
  %sub = sub nsw i32 0, %nx
  store i32 %sub, i32* %ptr, align 4
  br label %inner_loop_end

inner_loop_end:
  br i1 undef, label %inner_loop, label %middle_loop_end

middle_loop_end:
  br i1 undef, label %middle_loop, label %outer_loop_end

outer_loop_end:
  br label %outer_loop
}

; We have a diamond starting from %if, but %if.if is also reachable from %loop,
; so %gep should not be conditionally hoisted.
; CHECK-LABEL: @diamond_with_extra_in_edge
define void @diamond_with_extra_in_edge(i32* %ptr1, i32* %ptr2, i32 %arg) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp2 = icmp ne i32 0, %arg
; CHECK-DAG: %gep = getelementptr i32, i32* %ptr1, i32 4
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %phi1 = phi i32 [ 0, %entry ], [ %phi2, %then ]
  %cmp1 = icmp ugt i32 16, %phi1
  br i1 %cmp1, label %if, label %if.if

if:
  %cmp2 = icmp ne i32 0, %arg
  br i1 %cmp2, label %if.if, label %if.else

if.if:
  %gep = getelementptr i32, i32* %ptr1, i32 4
  %val = load i32, i32* %gep, align 4
  br label %then

if.else:
  br label %then

then:
  %phi2 = phi i32 [ %val, %if.if ], [ %phi1, %if.else ]
  store i32 %phi2, i32* %ptr2, align 4
  br label %loop
}

; %loop/%if/%then form a triangle, but %loop/%if/%then/%end also form a diamond.
; The triangle should be picked for conditional hoisting.
; CHECK-LABEL: @both_triangle_and_diamond
define void @both_triangle_and_diamond(i32* %ptr1, i32* %ptr2, i32 %arg) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 0, %arg
; CHECK-DAG: %gep = getelementptr i32, i32* %ptr1, i32 4
; CHECK: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK: [[IF_LICM]]:
; CHECK: br label %[[THEN_LICM]]

; CHECK: [[THEN_LICM]]:
; CHECK: %phi2 = phi i32 [ 0, %[[IF_LICM]] ], [ 1, %entry ]
; CHECK: br label %loop

loop:
  %phi1 = phi i32 [ 0, %entry ], [ %phi3, %end ]
  %cmp1 = icmp ne i32 0, %arg
  br i1 %cmp1, label %if, label %then

if:
  %gep = getelementptr i32, i32* %ptr1, i32 4
  %val = load i32, i32* %gep, align 4
  %cmp2 = icmp ugt i32 16, %phi1
  br i1 %cmp2, label %end, label %then

then:
  %phi2 = phi i32 [ 0, %if ], [ 1, %loop ]
  br label %end

end:
  %phi3 = phi i32 [ %phi2, %then ], [ %val, %if ]
  store i32 %phi3, i32* %ptr2, align 4
  br label %loop
}

; We shouldn't duplicate the branch at the end of %loop and should instead hoist
; %val to %entry.
; CHECK-LABEL: @same_destination_branch
define i32 @same_destination_branch(i32 %arg1, i32 %arg2) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 %arg2, 0
; CHECK-DAG: %val = add i32 %arg1, 1
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %phi = phi i32 [ 0, %entry ], [ %add, %then ]
  %add = add i32 %phi, 1
  %cmp1 = icmp ne i32 %arg2, 0
  br i1 %cmp1, label %if, label %if

if:
  %val = add i32 %arg1, 1
  br label %then

then:
  %cmp2 = icmp ne i32 %val, %phi
  br i1 %cmp2, label %loop, label %end

end:
  ret i32 %val
}

; Diamond-like control flow but the left/right blocks actually have the same
; destinations.
; TODO: We could potentially hoist all of phi2-4, but currently only hoist phi2.
; CHECK-LABEL: @diamond_like_same_destinations
define i32 @diamond_like_same_destinations(i32 %arg1, i32 %arg2) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 %arg1, 0
; CHECK-DAG: %cmp2 = icmp ugt i32 %arg2, 1
; CHECK-DAG: %cmp3 = icmp ugt i32 %arg2, 2
; CHECK: br i1 %cmp1, label %[[LEFT1_LICM:.*]], label %[[RIGHT1_LICM:.*]]
entry:
  br label %loop

; CHECK: [[LEFT1_LICM]]:
; CHECK: br label %[[LEFT2_LICM:.*]]

; CHECK: [[RIGHT1_LICM]]:
; CHECK: br label %[[LEFT2_LICM]]

; CHECK: [[LEFT2_LICM]]:
; CHECK: %phi2 = phi i32 [ 0, %[[LEFT1_LICM]] ], [ 1, %[[RIGHT1_LICM]] ]
; CHECK: br label %loop

loop:
  %phi1 = phi i32 [ 0, %entry ], [ %add, %loopend ]
  %add = add i32 %phi1, 1
  %cmp1 = icmp ne i32 %arg1, 0
  br i1 %cmp1, label %left1, label %right1

left1:
  %cmp2 = icmp ugt i32 %arg2, 1
  br i1 %cmp2, label %left2, label %right2

right1:
  %cmp3 = icmp ugt i32 %arg2, 2
  br i1 %cmp3, label %left2, label %right2

left2:
  %phi2 = phi i32 [ 0, %left1 ], [ 1, %right1 ]
  br label %loopend

right2:
  %phi3 = phi i32 [ 2, %left1 ], [ 3, %right1 ]
  br label %loopend

loopend:
  %phi4 = phi i32 [ %phi2, %left2 ], [ %phi3, %right2 ]
  %cmp4 = icmp ne i32 %phi1, 32
  br i1 %cmp4, label %loop, label %end

end:
  ret i32 %phi4
}

; A phi with multiple incoming values for the same block due to a branch with
; two destinations that are actually the same. We can't hoist this.
; TODO: This could be hoisted by erasing one of the incoming values.
; CHECK-LABEL: @phi_multiple_values_same_block
define i32 @phi_multiple_values_same_block(i32 %arg) {
; CHECK-LABEL: entry:
; CHECK: %cmp = icmp sgt i32 %arg, 4
; CHECK-NOT: phi
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %cmp = icmp sgt i32 %arg, 4
  br i1 %cmp, label %if, label %then

if:
  br i1 undef, label %then, label %then

then:
  %phi = phi i32 [ %arg, %loop ], [ 1, %if ], [ 1, %if ]
  br i1 undef, label %exit, label %loop

exit:
  ret i32 %phi
}
