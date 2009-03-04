// RUN: clang -analyze -checker-cfref --analyzer-store=region --verify -fblocks %s

//---------------------------------------------------------------------------
// Test case 'checkaccess_union' differs for region store and basic store.
// The basic store doesn't reason about compound literals, so the code
// below won't fire an "uninitialized value" warning.
//---------------------------------------------------------------------------

// PR 2948 (testcase; crash on VisitLValue for union types)
// http://llvm.org/bugs/show_bug.cgi?id=2948

void checkaccess_union() {
  int ret = 0, status;
  if (((((__extension__ (((union {  // expected-warning {{ Branch condition evaluates to an uninitialized value.}}
    __typeof (status) __in; int __i;}
    )
    {
      .__in = (status)}
      ).__i))) & 0xff00) >> 8) == 1)
        ret = 1;
}


// Check our handling of fields being invalidated by function calls.
struct test2_struct { int x; int y; char* s; };
void test2_helper(struct test2_struct* p);

char test2() {
  struct test2_struct s;
  test2_help(&s);
  char *p = 0;
  
  if (s.x > 1) {
    if (s.s != 0) {
      p = "hello";
    }
  }
  
  if (s.x > 1) {
    if (s.s != 0) {
      return *p;
    }
  }

  return 'a';
}
