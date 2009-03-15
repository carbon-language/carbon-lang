// RUN: clang -dump-tokens %s &> %t &&
// RUN: grep "identifier '\$A'" %t
// RUN: clang -dump-tokens -x assembler-with-cpp %s &> %t &&
// RUN: grep "identifier 'A'" %t
// PR3808

$A
