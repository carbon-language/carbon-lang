// PR 1417

// RUN: %llvmgcc -xc  %s -S -o - | grep "struct.anon = type \{\}"

struct { } *X;
