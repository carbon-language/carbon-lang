// RUN: clang-cc -verify -analyze -checker-cfref -analyzer-store=region %s

struct tea_cheese { unsigned magic; };
typedef struct tea_cheese kernel_tea_cheese_t;
extern kernel_tea_cheese_t _wonky_gesticulate_cheese;

// This test case exercises the ElementRegion::getRValueType() logic.


void foo( void )
{
  kernel_tea_cheese_t *wonky = &_wonky_gesticulate_cheese;
  struct load_wine *cmd = (void*) &wonky[1];
  cmd = cmd;
  char *p = (void*) &wonky[1];
  *p = 1; 
  kernel_tea_cheese_t *q = &wonky[1];
  kernel_tea_cheese_t r = *q; // expected-warning{{out-of-bound memory position}}
}
