; RUN: llc < %s -march=c

%A = type { i32, i8*, { i32, i32, i32, i32, i32, i32, i32, i32 }*, i16 }

define void @test(%A*) {
        ret void
}

