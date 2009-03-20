// RUN: clang -x assembler-with-cpp -E %s > %t &&

#ifndef __ASSEMBLER__
#error "__ASSEMBLER__ not defined"
#endif


// Invalid token pasting is ok. 
// RUN: grep '1: X .' %t &&
#define A X ## .
1: A

// Line markers are not linemarkers in .S files, they are passed through.
// RUN: grep '# 321' %t &&
# 321

// Unknown directives are passed through.
// RUN: grep '# B C' %t &&
# B C

// Unknown directives are expanded.
// RUN: grep '# BAR42' %t &&
#define D(x) BAR ## x
# D(42)

// Unmatched quotes are permitted.
// RUN: grep "2: '" %t &&
// RUN: grep '3: "' %t &&
2: '
3: "

// Empty char literals are ok.
// RUN: grep "4: ''" %t &&
4: ''

// RUN: true
