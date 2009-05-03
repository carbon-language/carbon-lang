// RUN: clang -E -o %t.1 %s &&
// RUN: clang -E -MD -MF %t.d -MT foo -o %t.2 %s &&
// RUN: diff %t.1 %t.2 &&
// RUN: grep "foo:" %t.d &&
// RUN: grep "dependencies-and-pp.c" %t.d
