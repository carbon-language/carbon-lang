; Test that upgrading shift instructions and constant expressions works
; correctly.
; RUN: llvm-upgrade < %s | grep {ashr i32 .X, 2}
; RUN: llvm-upgrade < %s | grep {lshr i32 .X, 2} 
; RUN: llvm-upgrade < %s | grep {shl i32 .X, 2}
; RUN: llvm-upgrade < %s | grep {ashr i32 .X, 6}
; RUN: llvm-upgrade < %s | grep {lshr i32 .X, 1}
; RUN: llvm-upgrade < %s | grep {shl i32 .X, 1}

void %test(int %X) {
  %A = ashr int %X, ubyte 2
  %B = lshr int %X, ubyte 2
  %C = shl  int %X, ubyte 2
  %D = ashr int %X, ubyte trunc ( int shl  (int 3, ubyte 1) to ubyte )
  %E = lshr int %X, ubyte trunc ( int ashr (int 3, ubyte 1) to ubyte )
  %F = shl  int %X, ubyte trunc ( int lshr (int 3, ubyte 1) to ubyte )
  ret void
}
