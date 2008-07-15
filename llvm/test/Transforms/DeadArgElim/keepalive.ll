; RUN: llvm-as < %s | opt -deadargelim | llvm-dis > %t
; RUN: cat %t | grep {define internal \{  \} @test}
; RUN: cat %t | grep {define internal \{ i32 \} @test}
; RUN: cat %t | grep {define internal \<\{ i32, i32 \}\> @test}

; Check if the pass doesn't modify anything that doesn't need changing. We feed
; an unused argument to each function to lure it into changing _something_ about
; the function and then changing too much.


; This checks if the struct retval isn't changed into a void
define internal {  } @test(i32 %DEADARG1) {
        ret {  } {  } 
}

; This checks if the struct retval isn't removed
define internal {i32} @test1(i32 %DEADARG1) {
        ret { i32 } { i32 1 } 
}

; This checks if the struct doesn't get non-packed
define internal <{ i32, i32 }> @test2(i32 %DEADARG1) {
        ret <{ i32, i32 }> <{ i32 1, i32 2 }>
}

; We use this external function to make sure the return values don't become dead
declare void @user({  }, { i32 }, <{ i32, i32 }>)

define void @caller() {
        %A = call {  } @test(i32 0)
        %B = call { i32 } @test1(i32 1)
        %C = call <{ i32, i32 }> @test2(i32 2)
        call void @user({  } %A, { i32 } %B, <{ i32, i32 }> %C)
        ret void
}

