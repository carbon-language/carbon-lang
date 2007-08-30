// RUN: clang -parse-ast-check %s

#define offsetof(TYPE, MEMBER) __builtin_offsetof (TYPE, MEMBER)

typedef struct P { int i; float f; } PT;
struct external_sun3_core
{
 unsigned c_regs; 

  PT  X[100];
  
};

void swap()
{
  int x;
  x = offsetof(struct external_sun3_core, c_regs);
  x = __builtin_offsetof(struct external_sun3_core, X[42].f);
  
  x = __builtin_offsetof(struct external_sun3_core, X[42].f2);  // expected-error {{no member named 'f2'}}
  x = __builtin_offsetof(int, X[42].f2);  // expected-error {{offsetof requires struct}}
}    

