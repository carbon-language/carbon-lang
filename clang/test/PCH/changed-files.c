const char *s0 = m0;
int s1 = m1;

// RUN: echo '#define m0 ""' > %t.h
// RUN: %clang_cc1 -emit-pch -o %t.h.pch %t.h
// RUN: echo '' > %t.h
// RUN: not %clang_cc1 -include-pch %t.h.pch %s 2>&1 | grep "size of file"

// RUN: echo '#define m0 000' > %t.h
// RUN: %clang_cc1 -emit-pch -o %t.h.pch %t.h
// RUN: echo '' > %t.h
// RUN: not %clang_cc1 -include-pch %t.h.pch %s 2>&1 | grep "size of file"

// RUN: echo '#define m0 000' > %t.h
// RUN: echo "#define m1 'abcd'" >> %t.h
// RUN: %clang_cc1 -emit-pch -o %t.h.pch %t.h
// RUN: echo '' > %t.h
// RUN: not %clang_cc1 -include-pch %t.h.pch %s 2>&1 | grep "size of file"
