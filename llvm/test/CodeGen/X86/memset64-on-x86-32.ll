; RUN: llc < %s -mtriple=i386-apple-darwin   -mcpu=nehalem | grep movups | count 5
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2   | grep movups   | count 5

define void @bork() nounwind {
entry:
        call void @llvm.memset.i64( i8* null, i8 0, i64 80, i32 4 )
        ret void
}

declare void @llvm.memset.i64(i8*, i8, i64, i32) nounwind

