; RUN: llc < %s -march=c

%X = type { i32, float }

define void @test() {
        getelementptr %X* null, i64 0, i32 1            ; <float*>:1 [#uses=0]
        ret void
}

