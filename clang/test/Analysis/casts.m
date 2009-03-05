// RUN: clang -analyze -checker-cfref -analyzer-store=basic --verify %s &&
// RUN: clang -analyze -checker-cfref -analyzer-store=region --verify %s

// Test function pointer casts.  Currently we track function addresses using
// loc::FunctionVal.  Because casts can be arbitrary, do we need to model
// functions with regions?

typedef void (*MyFuncTest1)(void);

MyFuncTest1 test1_aux(void);
void test1(void) {
  void *x;
  void* (*p)(void);
  p = ((void*) test1_aux());
  if (p != ((void*) 0)) x = (*p)();
}
