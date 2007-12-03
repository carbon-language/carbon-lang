// RUN: clang -arch ppc -arch bogusW16W16 -fsyntax-only %s 2>&1 | grep note | wc -l | grep 1

// wchar_t varies across targets.
void *X = L"foo";

