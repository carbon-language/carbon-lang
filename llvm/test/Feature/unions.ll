; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%union.anon = type union { i8, i32, float }

@union1 = constant union { i32, i8 } { i32 4 }
@union2 = constant union { i32, i8 } insertvalue(union { i32, i8 } undef, i32 4, 0)

define void @"Unions" () {
  ret void
}
