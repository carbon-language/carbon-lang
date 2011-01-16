; RUN: opt -constmerge %s -S -o - | FileCheck %s
; Test that in one run var3 is merged into var2 and var1 into var4.

declare void @zed(%struct.foobar*, %struct.foobar*)

%struct.foobar = type { i32 }

@var1 = internal constant %struct.foobar { i32 2 }
@var2 = unnamed_addr constant %struct.foobar { i32 2 }
@var3 = internal constant %struct.foobar { i32 2 }
@var4 = unnamed_addr constant %struct.foobar { i32 2 }

; CHECK:      %struct.foobar = type { i32 }
; CHECK-NOT: @
; CHECK: @var2 = constant %struct.foobar { i32 2 }
; CHECK-NEXT: @var4 = constant %struct.foobar { i32 2 }
; CHECK-NOT: @
; CHECK: declare void @zed(%struct.foobar*, %struct.foobar*)

define i32 @main() {
entry:
  call void @zed(%struct.foobar* @var1, %struct.foobar* @var2)
  call void @zed(%struct.foobar* @var3, %struct.foobar* @var4)
  ret i32 0
}

