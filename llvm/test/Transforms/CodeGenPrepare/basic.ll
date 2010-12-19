; RUN: opt -codegenprepare %s -S | FileCheck %s
; PR8642

%0 = type <{ %1, %1 }>
%1 = type { i8, i8, i8, i8 }

@g_2 = global %0 <{ %1 { i8 1, i8 0, i8 0, i8 undef }, %1 { i8 2, i8 0, i8 0, i8 undef } }>, align 4
@g_4 = global %1 { i8 3, i8 0, i8 0, i8 undef }, align 4

; CGP shouldn't fold away the empty cond.false.i block, because the constant
; expr that will get dropped into it could trap.
define i16 @test1(i8** %argv, i1 %c) nounwind ssp {
entry:
  br i1 %c, label %cond.end.i, label %cond.false.i

cond.false.i:                                     ; preds = %entry
  br label %foo.exit

cond.end.i:                                       ; preds = %entry
  store i8* null, i8** %argv
  br label %foo.exit

foo.exit:                                         ; preds = %cond.end.i, %cond.false.i
  %call1 = phi i16 [ trunc (i32 srem (i32 1, i32 zext (i1 icmp eq (%1* bitcast (i8* getelementptr inbounds (%0* @g_2, i64 0, i32 1, i32 0) to %1*), %1* @g_4) to i32)) to i16), %cond.false.i ], [ 1, %cond.end.i ]
  ret i16 %call1
  
; CHECK: @test1
; CHECK: cond.false.i:
; CHECK-NEXT:  br label %foo.exit
}

