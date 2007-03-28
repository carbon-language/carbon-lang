; RUN: llvm-as < %s | llc -march=x86-64 | grep 'leal (%rdi,%rdi,2), %eax'

define i32 @test(i32 %a) {
        %tmp2 = mul i32 %a, 3           ; <i32> [#uses=1]
        ret i32 %tmp2
}

