; Test function notes
; RUN: llvm-as < %s -f -o /dev/null

define void @fn1() notes(inline=always) {
  ret void
}

define void @fn2() notes(inline=never) {
  ret void
}

