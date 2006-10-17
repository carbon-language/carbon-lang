// RUN: clang -fsyntax-only %s

static __inline void __attribute__((__always_inline__, __nodebug__))
foo (void)
{
}
