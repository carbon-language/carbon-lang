// rdar://6533411
// RUN: clang -MD -MF %t.d -c -x c -o %t.o /dev/null && 

// RUN: grep '.*dependency-gen.c.out.tmp.o:' %t.d
// RUN: grep '/dev/null' %t.d
