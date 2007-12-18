// RUN: clang %s -fsyntax-only

const char* test1 = 1 ? "i" : 1 == 1 ? "v" : "r";

