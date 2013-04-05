// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,unix.MismatchedDeallocator,cplusplus.NewDelete -std=c++11 -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,unix.MismatchedDeallocator,cplusplus.NewDelete,alpha.cplusplus.NewDeleteLeaks -DLEAKS -std=c++11 -verify %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

//--------------------------------------------------
// Check that unix.Malloc catches all types of bugs.
//--------------------------------------------------
void testMallocDoubleFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  free(p); // expected-warning{{Attempt to free released memory}}
}

void testMallocLeak() {
  int *p = (int *)malloc(sizeof(int));
} // expected-warning{{Memory is never released; potential leak of memory pointed to by 'p'}}

void testMallocUseAfterFree() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  int j = *p; // expected-warning{{Use of memory after it is freed}}
}

void testMallocBadFree() {
  int i;
  free(&i); // expected-warning{{Argument to free() is the address of the local variable 'i', which is not memory allocated by malloc()}}
}

void testMallocOffsetFree() {
  int *p = (int *)malloc(sizeof(int));
  free(++p); // expected-warning{{Argument to free() is offset by 4 bytes from the start of memory allocated by malloc()}}
}

//-----------------------------------------------------------------
// Check that unix.MismatchedDeallocator catches all types of bugs.
//-----------------------------------------------------------------
void testMismatchedDeallocator() {
  int *x = (int *)malloc(sizeof(int));
  delete x; // expected-warning{{Memory allocated by malloc() should be deallocated by free(), not 'delete'}}
}

//----------------------------------------------------------------
// Check that alpha.cplusplus.NewDelete catches all types of bugs.
//----------------------------------------------------------------
void testNewDoubleFree() {
  int *p = new int;
  delete p;
  delete p; // expected-warning{{Attempt to free released memory}}
}

void testNewLeak() {
  int *p = new int;
}
#ifdef LEAKS
// expected-warning@-2 {{Memory is never released; potential leak of memory pointed to by 'p'}}
#endif

void testNewUseAfterFree() {
  int *p = (int *)operator new(0);
  delete p;
  int j = *p; // expected-warning{{Use of memory after it is freed}}
}

void testNewBadFree() {
  int i;
  delete &i; // expected-warning{{Argument to 'delete' is the address of the local variable 'i', which is not memory allocated by 'new'}}
}

void testNewOffsetFree() {
  int *p = new int;
  operator delete(++p); // expected-warning{{Argument to operator delete is offset by 4 bytes from the start of memory allocated by 'new'}}
}
