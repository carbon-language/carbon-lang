// PR 1419

// RUN: %llvmgcc -xc  %s -c -o - | llvm-dis | grep "struct.anon = type \{  \}"

struct { } *X;
