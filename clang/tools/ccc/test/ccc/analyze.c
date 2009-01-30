// RUN: xcc --analyze %s -o %t &&
// RUN: grep '<string>Dereference of null pointer.</string>' %t &&

// RUN: xcc -### --analyze %s -Xanalyzer -check-that-program-halts &> %t &&
// RUN: grep 'check-that-program-halts' %t

void f(int *p) {
  if (!p) 
    *p = 0;
}
