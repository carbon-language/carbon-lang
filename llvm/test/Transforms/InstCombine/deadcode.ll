; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep {ret i32 %A}

int %test(int %A) {
  %X = or bool false, false
  br bool %X, label %T, label %C
T:
  %B = add int %A, 1
  br label %C
C:
  %C = phi int [%B, %T], [%A, %0]
  ret int %C
}
