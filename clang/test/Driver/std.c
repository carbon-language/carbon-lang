// RUN: clang -std=c99 -trigraphs -std=gnu99 %s -E -o %t &&
// RUN: grep '??(??)' %t &&
// RUN: clang -ansi %s -E -o %t &&
// RUN: grep -F '[]' %t &&
// RUN: clang -std=gnu99 -trigraphs %s -E -o %t &&
// RUN: grep -F '[]' %t

??(??)
