; PR1187
; RUN: llvm-upgrade < %s > /dev/null

implementation 
internal void %func(int %x) {
  ret void
}

internal void %func(uint %x) {
  ret void
}
