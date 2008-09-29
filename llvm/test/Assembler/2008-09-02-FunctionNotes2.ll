; Test function notes
; RUN: not llvm-as  %s |& grep "only one inline note" 
; XFAIL: *
define void @fn1() alwaysinline  noinline {
  ret void
}

