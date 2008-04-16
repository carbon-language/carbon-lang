; RUN: llvm-as < %s |   opt -adce | llvm-dis | grep switch
; PR 1564
; XFAIL: *
  
define fastcc void @out() {
    start:
            br label %loop
    unreachable:
            unreachable
    loop:
            switch i32 0, label %unreachable [
                     i32 0, label %loop
            ]
}
