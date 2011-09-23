// RUN: touch %t.h
// RUN: echo '#include "%t.h"' > %t.c
// RUN: %clang --working-directory %t -fsyntax-only %t.c
