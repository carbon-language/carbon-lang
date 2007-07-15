// RUN: clang %s -fsyntax-only

char array[1024/(sizeof (long))];

int x['\xBb' == (char) 187 ? 1: -1];

