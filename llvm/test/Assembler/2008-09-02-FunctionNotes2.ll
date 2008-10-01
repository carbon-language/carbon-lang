; Test function notes
; RUN: not llvm-as %s -o /dev/null -f |& grep "only one inline note" 
; XFAIL: *
define void @fn1() alwaysinline  noinline {
  ret void
}

