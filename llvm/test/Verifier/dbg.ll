; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s

define void @foo() {
  ret void, !dbg !{}
}

; CHECK: invalid !dbg metadata attachment
; CHECK-NEXT: ret void, !dbg ![[LOC:[0-9]+]]
; CHECK-NEXT: ![[LOC]] = !{}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
