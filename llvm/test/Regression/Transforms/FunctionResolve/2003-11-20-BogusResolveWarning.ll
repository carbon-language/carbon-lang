; RUN: llvm-as < %s | opt -funcresolve -disable-output 2>&1 | not grep WARNING


void %test() {
  call int(...)* %test()
  ret void
}

declare int %test(...)

