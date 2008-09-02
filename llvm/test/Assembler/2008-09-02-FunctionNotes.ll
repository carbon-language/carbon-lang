; Test function notes
; RUN: llvm-as < %s | llvm-dis | grep inline | count 2

define void @fn1() notes(inline=always) {
  ret void
}

define void @fn2() notes(inline=never) {
  ret void
}

define void @fn3() {
  ret void
}
