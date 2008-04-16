; RUN: llvm-as < %s | llc -march=x86 | grep stosb

target triple = "i386-apple-darwin9"
        %struct.S = type { [80 x i8] }

define %struct.S* @bork() {
entry:
        call void @llvm.memset.i64( i8* null, i8 0, i64 80, i32 1 )
        ret %struct.S* null
}

declare void @llvm.memset.i64(i8*, i8, i64, i32) nounwind

