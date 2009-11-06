// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s
// PR3135

typedef struct {
  int *a;
} structure;

int bar(structure *x);

int foo()
{
  int x;
  structure y = {&x};

  // the call to bar may initialize x
  if (bar(&y) && x) // no-warning
    return 1;

  return 0;
}
