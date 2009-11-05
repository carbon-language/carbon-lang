// rdar://6533411
// RUN: clang -MD -MF %t.d -S -x c -o %t.o %s && 
// RUN: grep '.*dependency-gen.*:' %t.d &&
// RUN: grep 'dependency-gen.c' %t.d &&

// RUN: clang -S -M -x c %s -o %t.d &&
// RUN: grep '.*dependency-gen.*:' %t.d &&
// RUN: grep 'dependency-gen.c' %t.d
