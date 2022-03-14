; RUN: llvm-as < %s -o /dev/null

@a = global i32 42
@llvm.used = appending global [1 x i32*] [i32* @a], section "llvm.metadata"
