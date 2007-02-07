; PR1187
; RUN: llvm-upgrade < %s > /dev/null

implementation 

i1 %func(i8 %x, i16 %x, i32 %x, i64 %x) {
  ret void
}
