; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i1 false}
; PR2329

define i1 @f() {
  %x = icmp eq i8* inttoptr (i32 1 to i8*), inttoptr (i32 2 to i8*)
  ret i1 %x
}

