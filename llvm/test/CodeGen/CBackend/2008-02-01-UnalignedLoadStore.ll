; RUN: llc < %s -march=c | \
; RUN:          grep {struct __attribute__ ((packed, aligned(} | count 4

define void @test(i32* %P) {
        %X = load i32* %P, align 1
        store i32 %X, i32* %P, align 1
        ret void
}

define void @test2(i32* %P) {
        %X = load volatile i32* %P, align 2
        store volatile i32 %X, i32* %P, align 2
        ret void
}

