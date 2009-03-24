// RUN: clang-cc -analyze -checker-cfref --analyzer-store=basic --verify -fblocks %s

//---------------------------------------------------------------------------
// Test case 'checkaccess_union' differs for region store and basic store.
// The basic store doesn't reason about compound literals, so the code
// below won't fire an "uninitialized value" warning.
//---------------------------------------------------------------------------

// PR 2948 (testcase; crash on VisitLValue for union types)
// http://llvm.org/bugs/show_bug.cgi?id=2948

void checkaccess_union() {
  int ret = 0, status;
  if (((((__extension__ (((union {  // no-warning
    __typeof (status) __in; int __i;}
    )
    {
      .__in = (status)}
      ).__i))) & 0xff00) >> 8) == 1)
        ret = 1;
}
