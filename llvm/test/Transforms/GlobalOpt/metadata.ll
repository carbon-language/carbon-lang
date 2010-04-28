; RUN: opt -S -globalopt < %s | FileCheck %s

; PR6112 - When globalopt does RAUW(@G, %G), the metadata reference should drop
; to null.
@G = internal global i8** null

define i32 @main(i32 %argc, i8** %argv) {
; CHECK: @main
; CHECK: %G = alloca
  store i8** %argv, i8*** @G
  ret i32 0
}

!named = !{!0}

; CHECK: !0 = metadata !{null}
!0 = metadata !{i8*** @G}


