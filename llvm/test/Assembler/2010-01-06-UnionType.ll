; RUN: llvm-as %s -o /dev/null

%X = type union { i32, i32* }
