; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crashcalls -silence-passes > /dev/null
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck %s
; REQUIRES: loadable_module

; Bugpoint should keep the call's metadata attached to the call.

; CHECK: call void @foo(), !dbg ![[LOC:[0-9]+]], !attach ![[CALL:[0-9]+]]
; CHECK: ![[LOC]] = !{i32 104, i32 105, ![[SCOPE:[0-9]+]], ![[SCOPE]]}
; CHECK: ![[SCOPE]] = !{!"0x11\000\00me\001\00\000\00\000", ![[FILE:[0-9]+]], ![[LIST:[0-9]+]], ![[LIST]], null, null, null}
; CHECK: ![[FILE]] = !{!"source.c", !"/dir"}
; CHECK: ![[LIST]] = !{i32 0}
; CHECK: ![[CALL]] = !{!"the call to foo"}

%rust_task = type {}
define void @test(i32* %a, i8* %b) {
    %s = mul i8 22, 9, !attach !0, !dbg !10
    store i8 %s, i8* %b, !attach !1, !dbg !11
    call void @foo(), !attach !2, !dbg !12
    store i32 7, i32* %a, !attach !3, !dbg !13
    %t = add i32 0, 5, !attach !4, !dbg !14
    ret void
}

declare void @foo()

!llvm.module.flags = !{!17}

!0 = !{!"boring"}
!1 = !{!"uninteresting"}
!2 = !{!"the call to foo"}
!3 = !{!"noise"}
!4 = !{!"filler"}

!9 = !{!"0x11\000\00me\001\00\000\00\000", !15, !16, !16, null, null, null} ; [ DW_TAG_compile_unit ]
!10 = !{i32 100, i32 101, !9, !9}
!11 = !{i32 102, i32 103, !9, !9}
!12 = !{i32 104, i32 105, !9, !9}
!13 = !{i32 106, i32 107, !9, !9}
!14 = !{i32 108, i32 109, !9, !9}
!15 = !{!"source.c", !"/dir"}
!16 = !{i32 0}
!17 = !{i32 1, !"Debug Info Version", i32 2}
