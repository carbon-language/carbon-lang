; RUN: opt -S < %s -jump-threading | FileCheck %s

; Keep block addresses alive.
@addresses = constant [4 x i8*] [
  i8* blockaddress(@test1, %L1), i8* blockaddress(@test1, %L2),
  i8* blockaddress(@test2, %L1), i8* blockaddress(@test2, %L2)
]

declare void @bar()
declare void @baz()



; Check basic jump threading for indirectbr instructions.

; CHECK: void @test1
; CHECK: br i1 %tobool, label %L1, label %indirectgoto
; CHECK-NOT: if.else:
; CHECK: L1:
; CHECK: indirectbr i8* %address, [label %L1, label %L2]
define void @test1(i32 %i, i8* %address) nounwind {
entry:
  %rem = srem i32 %i, 2
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %indirectgoto, label %if.else

if.else:                                          ; preds = %entry
  br label %indirectgoto

L1:                                               ; preds = %indirectgoto
  call void @bar()
  ret void

L2:                                               ; preds = %indirectgoto
  call void @baz()
  ret void

indirectgoto:                                     ; preds = %if.else, %entry
  %indirect.goto.dest = phi i8* [ %address, %if.else ], [ blockaddress(@test1, %L1), %entry ]
  indirectbr i8* %indirect.goto.dest, [label %L1, label %L2]
}


; Check constant folding of indirectbr

; CHECK: void @test2
; CHECK-NEXT: :
; CHECK-NEXT: call void @bar
; CHECK-NEXT: ret void
define void @test2() nounwind {
entry:
  indirectbr i8* blockaddress(@test2, %L1), [label %L1, label %L2]

L1:                                               ; preds = %indirectgoto
  call void @bar()
  ret void

L2:                                               ; preds = %indirectgoto
  call void @baz()
  ret void
}
