// rdar://6533411
// RUN: clang -MD -MF %t.d -c -x c -o %t.o /dev/null && 
// RUN: grep '.*dependency-gen.*:' %t.d &&
// RUN: grep '/dev/null' %t.d &&

// RUN: clang -M -x c /dev/null -o %t.deps &&
// RUN: grep 'null.o: /dev/null' %t.deps
