; RUN: opt < %s -simplifycfg -S | FileCheck %s

declare void @foo1()

declare void @foo2()

define void @test1(i32 %V) {
        %C1 = icmp eq i32 %V, 4         ; <i1> [#uses=1]
        %C2 = icmp eq i32 %V, 17                ; <i1> [#uses=1]
        %CN = or i1 %C1, %C2            ; <i1> [#uses=1]
        br i1 %CN, label %T, label %F
T:              ; preds = %0
        call void @foo1( )
        ret void
F:              ; preds = %0
        call void @foo2( )
        ret void
; CHECK: @test1
; CHECK:  switch i32 %V, label %F [
; CHECK:    i32 17, label %T
; CHECK:    i32 4, label %T
; CHECK:  ]
}

define void @test2(i32 %V) {
        %C1 = icmp ne i32 %V, 4         ; <i1> [#uses=1]
        %C2 = icmp ne i32 %V, 17                ; <i1> [#uses=1]
        %CN = and i1 %C1, %C2           ; <i1> [#uses=1]
        br i1 %CN, label %T, label %F
T:              ; preds = %0
        call void @foo1( )
        ret void
F:              ; preds = %0
        call void @foo2( )
        ret void
; CHECK: @test2
; CHECK:  switch i32 %V, label %T [
; CHECK:    i32 17, label %F
; CHECK:    i32 4, label %F
; CHECK:  ]
}

define void @test3(i32 %V) {
        %C1 = icmp eq i32 %V, 4         ; <i1> [#uses=1]
        br i1 %C1, label %T, label %N
N:              ; preds = %0
        %C2 = icmp eq i32 %V, 17                ; <i1> [#uses=1]
        br i1 %C2, label %T, label %F
T:              ; preds = %N, %0
        call void @foo1( )
        ret void
F:              ; preds = %N
        call void @foo2( )
        ret void

; CHECK: @test3
; CHECK: switch i32 %V, label %F [
; CHECK:     i32 4, label %T
; CHECK:     i32 17, label %T
; CHECK:   ]
}



define i32 @test4(i8 zeroext %c) nounwind ssp noredzone {
entry:
  %cmp = icmp eq i8 %c, 62
  br i1 %cmp, label %lor.end, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %cmp4 = icmp eq i8 %c, 34
  br i1 %cmp4, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %lor.lhs.false
  %cmp8 = icmp eq i8 %c, 92
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %lor.lhs.false, %entry
  %0 = phi i1 [ true, %lor.lhs.false ], [ true, %entry ], [ %cmp8, %lor.rhs ]
  %lor.ext = zext i1 %0 to i32
  ret i32 %lor.ext
  
; CHECK: @test4
; CHECK:  switch i8 %c, label %lor.rhs [
; CHECK:    i8 62, label %lor.end
; CHECK:    i8 34, label %lor.end
; CHECK:    i8 92, label %lor.end
; CHECK:  ]
}

define i32 @test5(i8 zeroext %c) nounwind ssp noredzone {
entry:
  switch i8 %c, label %lor.rhs [
    i8 62, label %lor.end
    i8 34, label %lor.end
    i8 92, label %lor.end
  ]

lor.rhs:                                          ; preds = %entry
  %V = icmp eq i8 %c, 92
  br label %lor.end

lor.end:                                          ; preds = %entry, %entry, %entry, %lor.rhs
  %0 = phi i1 [ true, %entry ], [ %V, %lor.rhs ], [ true, %entry ], [ true, %entry ]
  %lor.ext = zext i1 %0 to i32
  ret i32 %lor.ext
; CHECK: @test5
; CHECK:  switch i8 %c, label %lor.rhs [
; CHECK:    i8 62, label %lor.end
; CHECK:    i8 34, label %lor.end
; CHECK:    i8 92, label %lor.end
; CHECK:  ]
}


define i1 @test6({ i32, i32 }* %I) {
entry:
        %tmp.1.i = getelementptr { i32, i32 }* %I, i64 0, i32 1         ; <i32*> [#uses=1]
        %tmp.2.i = load i32* %tmp.1.i           ; <i32> [#uses=6]
        %tmp.2 = icmp eq i32 %tmp.2.i, 14               ; <i1> [#uses=1]
        br i1 %tmp.2, label %shortcirc_done.4, label %shortcirc_next.0
shortcirc_next.0:               ; preds = %entry
        %tmp.6 = icmp eq i32 %tmp.2.i, 15               ; <i1> [#uses=1]
        br i1 %tmp.6, label %shortcirc_done.4, label %shortcirc_next.1
shortcirc_next.1:               ; preds = %shortcirc_next.0
        %tmp.11 = icmp eq i32 %tmp.2.i, 16              ; <i1> [#uses=1]
        br i1 %tmp.11, label %shortcirc_done.4, label %shortcirc_next.2
shortcirc_next.2:               ; preds = %shortcirc_next.1
        %tmp.16 = icmp eq i32 %tmp.2.i, 17              ; <i1> [#uses=1]
        br i1 %tmp.16, label %shortcirc_done.4, label %shortcirc_next.3
shortcirc_next.3:               ; preds = %shortcirc_next.2
        %tmp.21 = icmp eq i32 %tmp.2.i, 18              ; <i1> [#uses=1]
        br i1 %tmp.21, label %shortcirc_done.4, label %shortcirc_next.4
shortcirc_next.4:               ; preds = %shortcirc_next.3
        %tmp.26 = icmp eq i32 %tmp.2.i, 19              ; <i1> [#uses=1]
        br label %UnifiedReturnBlock
shortcirc_done.4:               ; preds = %shortcirc_next.3, %shortcirc_next.2, %shortcirc_next.1, %shortcirc_next.0, %entry
        br label %UnifiedReturnBlock
UnifiedReturnBlock:             ; preds = %shortcirc_done.4, %shortcirc_next.4
        %UnifiedRetVal = phi i1 [ %tmp.26, %shortcirc_next.4 ], [ true, %shortcirc_done.4 ]             ; <i1> [#uses=1]
        ret i1 %UnifiedRetVal
        
; CHECK: @test6
; CHECK:   switch i32 %tmp.2.i, label %shortcirc_next.4 [
; CHECK:       i32 14, label %UnifiedReturnBlock
; CHECK:       i32 15, label %UnifiedReturnBlock
; CHECK:       i32 16, label %UnifiedReturnBlock
; CHECK:       i32 17, label %UnifiedReturnBlock
; CHECK:       i32 18, label %UnifiedReturnBlock
; CHECK:       i32 19, label %switch.edge
; CHECK:     ]

}


