; Test function notes
; RUN: not llvm-as  %s |& grep "only one inline note" 

define void @fn1() notes(inline=always,inline=never) {
  ret void
}

