// RUN: %llvmgcc -S -g -O %s -o - | llc
// RUN: %llvmgcc -S -g %s -o - | llc

extern void foo (void);

static
void baz (int i)
{
  foo ();
  typedef char A[i];
  struct { A b; } *x = 0;
}

void
bar (i)
{
  baz (i);
}
