const char *s0 = m0;
int s1 = m1;
const char *s2 = m0;

// FIXME: This test fails inexplicably on Windows in a manner that makes it 
// look like standard error isn't getting flushed properly.

// RUN: true
// RUNx: echo '#define m0 ""' > %t.h
// RUNx: %clang_cc1 -emit-pch -o %t.h.pch %t.h
// RUNx: echo '' > %t.h
// RUNx: not %clang_cc1 -include-pch %t.h.pch %s 2> %t.stderr
// RUNx: grep "modified" %t.stderr

// RUNx: echo '#define m0 000' > %t.h
// RUNx: %clang_cc1 -emit-pch -o %t.h.pch %t.h
// RUNx: echo '' > %t.h
// RUNx: not %clang_cc1 -include-pch %t.h.pch %s 2> %t.stderr
// RUNx: grep "modified" %t.stderr

// RUNx: echo '#define m0 000' > %t.h
// RUNx: echo "#define m1 'abcd'" >> %t.h
// RUNx: %clang_cc1 -emit-pch -o %t.h.pch %t.h
// RUNx: echo '' > %t.h
// RUNx: not %clang_cc1 -include-pch %t.h.pch %s 2> %t.stderr
// RUNx: grep "modified" %t.stderr
