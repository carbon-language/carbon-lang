// RUN: %llvmgcc %s -S -o /dev/null 2>&1 | not grep 'internal compiler error'

struct A X[(927 - 37) / sizeof(struct A)];
