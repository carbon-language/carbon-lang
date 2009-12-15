// RUN: %clang_cc1 -dump-tokens %s 2> %t
// RUN: grep "identifier '\$A'" %t
// RUN: %clang_cc1 -dump-tokens -x assembler-with-cpp %s 2> %t
// RUN: grep "identifier 'A'" %t
// PR3808

$A
