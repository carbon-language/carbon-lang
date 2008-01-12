// RUN: clang -fsyntax-only %s

int abcdefghi12(void) {
 const char (*ss)[12] = &__func__;
 return sizeof(__func__);
}

