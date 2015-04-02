; RUN: lli -jit-kind=orc-lazy %s

define private void @_ZL3foov() {
entry:
  ret void
}

define i32 @main(i32 %argc, i8** nocapture readnone %argv) {
entry:
  tail call void @_ZL3foov()
  ret i32 0
}
