; RUN: not llvm-as -f %s -o /dev/null


%T = type { int }

declare %T %test()

