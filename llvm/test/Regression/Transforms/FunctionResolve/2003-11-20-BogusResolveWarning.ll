; RUN: llvm-upgrade < %s | llvm-as | opt -funcresolve -disable-output 2>&1 | not grep WARNING


void %test() {
  call int(...)* %test()
  ret void
}

declare int %test(...)

