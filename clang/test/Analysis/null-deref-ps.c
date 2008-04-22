// RUN: clang -checker-simple -verify %s

void f1(int *p) {  
  if (p) *p = 1;
  else *p = 0; // expected-warning{{ereference}}
}

struct foo_struct {
  int x;
};

int f2(struct foo_struct* p) {
  
  if (p)
    p->x = 1;
    
  return p->x++; // expected-warning{{Dereference of null pointer.}}
}

int f3(char* x) {
  
  int i = 2;
  
  if (x)
    return x[i - 1];
  
  return x[i+1]; // expected-warning{{Dereference of null pointer.}}
}

