// RUN: clang-cc -dump-tokens %s 2> %t &&
// RUN: grep "identifier '\$A'" %t
// RUN: clang-cc -dump-tokens -x assembler-with-cpp %s 2> %t &&
// RUN: grep "identifier 'A'" %t
// PR3808

$A
