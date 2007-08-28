// RUN: clang -parse-ast-check -pedantic %s

struct s; 
struct s* t (struct s z[]) {   // expected-error {{array has incomplete element type}}
  return z;
}

void ff() { 
  struct s v, *p; // expected-error {{variable has incomplete type 'struct s'}}

  p = &v;
}

void *k (void l[2]) {          // expected-error {{array has incomplete element type}}
  return l; 
}

struct vari {
  int a;
  int b[];
};

struct vari *func(struct vari a[]) { // expected-error {{'struct vari' may not be used as an array element due to flexible array member}}
  return a;
}

int foo[](void);  // expected-error {{'foo' declared as array of functions}}

typedef int (*pfunc)(void);

pfunc xx(int f[](void)) { // expected-error {{'f' declared as array of functions}}
  return f;
}
