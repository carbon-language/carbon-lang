// RUN: %clang_cc1 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -verify -Warc-abi -fblocks -triple x86_64-apple-darwin10.0.0 %s

typedef __strong id strong_id;
typedef __weak id weak_id;
void test_pseudo_destructors(__strong id *sptr, __weak id *wptr) {
  sptr->~id(); // okay
  wptr->~id(); // okay
  sptr->~strong_id(); // okay
  wptr->~weak_id();
  sptr->~weak_id(); // expected-error{{pseudo-destructor destroys object of type '__strong id' with inconsistently-qualified type 'weak_id' (aka '__weak id')}}
  wptr->strong_id::~strong_id(); // expected-error{{pseudo-destructor destroys object of type '__weak id' with inconsistently-qualified type 'strong_id' (aka '__strong id')}}
  
  sptr->id::~id(); // okay
  wptr->id::~id(); // okay
}

void test_delete(__strong id *sptr, __weak id *wptr) {
  delete sptr;
  delete wptr;
  delete [] sptr; // expected-warning{{destroying an array of '__strong id'; this array must not have been allocated from non-ARC code}}
  delete [] wptr; // expected-warning{{destroying an array of '__weak id'; this array must not have been allocated from non-ARC code}}
}

void test_new(int n) {
  (void)new strong_id;
  (void)new weak_id;
  (void)new strong_id [n]; // expected-warning{{allocating an array of 'strong_id' (aka '__strong id'); this array must not be deleted in non-ARC code}}
  (void)new weak_id [n]; // expected-warning{{allocating an array of 'weak_id' (aka '__weak id'); this array must not be deleted in non-ARC code}}

  (void)new __strong id;
  (void)new __weak id;
  (void)new __strong id [n]; // expected-warning{{allocating an array of '__strong id'; this array must not be deleted in non-ARC code}}

  // Infer '__strong'.
  __strong id *idptr = new id;
  __strong id *idptr2 = new id [n]; // expected-warning{{allocating an array of '__strong id'; this array must not be deleted in non-ARC code}}

  // ... but not for arrays.
  typedef id id_array[2][3];
  (void)new id_array; // expected-error{{'new' cannot allocate an array of 'id' with no explicit lifetime}}

  typedef __strong id strong_id_array[2][3];
  typedef __strong id strong_id_3[3];
  strong_id_3 *idptr3 = new strong_id_array; // expected-warning{{allocating an array of '__strong id'; this array must not be deleted in non-ARC code}}
}

void test_jump_scope() {
  goto done; // expected-error{{goto into protected scope}}
  __strong id x; // expected-note{{jump bypasses initialization of retaining variable}}
 done:
  return;
}
