; RUN: llvm-upgrade < %s | not llvm-as &| grep {not verify as correct}
; PR1042

int %foo() {
   %A = invoke int %foo( )
        to label %L unwind label %L             ; <int> [#uses=1]

L: ; preds = %0, %0
   ret int %A
}
