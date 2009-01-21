// RUN: xcc --analyze %s -o %t &&
// RUN: grep '<string>Dereference of null pointer.</string>' %t

void f(int *p) {
  if (!p) 
    *p = 0;
}
