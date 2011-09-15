// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety %s

#define LOCKABLE            __attribute__ ((lockable))
#define SCOPED_LOCKABLE     __attribute__ ((scoped_lockable))
#define GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
#define GUARDED_VAR         __attribute__ ((guarded_var))
#define PT_GUARDED_BY(x)    __attribute__ ((pt_guarded_by(x)))
#define PT_GUARDED_VAR      __attribute__ ((pt_guarded_var))
#define ACQUIRED_AFTER(...) __attribute__ ((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...) __attribute__ ((acquired_before(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)   __attribute__ ((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)      __attribute__ ((shared_lock_function(__VA_ARGS__)))
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__ ((exclusive_trylock_function(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__ ((shared_trylock_function(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...)            __attribute__ ((unlock_function(__VA_ARGS__)))
#define LOCK_RETURNED(x)    __attribute__ ((lock_returned(x)))
#define LOCKS_EXCLUDED(...) __attribute__ ((locks_excluded(__VA_ARGS__)))
#define EXCLUSIVE_LOCKS_REQUIRED(...) \
  __attribute__ ((exclusive_locks_required(__VA_ARGS__)))
#define SHARED_LOCKS_REQUIRED(...) \
  __attribute__ ((shared_locks_required(__VA_ARGS__)))
#define NO_THREAD_SAFETY_ANALYSIS  __attribute__ ((no_thread_safety_analysis))

//-----------------------------------------//
//  Helper fields
//-----------------------------------------//


class  __attribute__((lockable)) Mutex {
 public:
  void Lock() __attribute__((exclusive_lock_function));
  void ReaderLock() __attribute__((shared_lock_function));
  void Unlock() __attribute__((unlock_function));
  bool TryLock() __attribute__((exclusive_trylock_function(true)));
  bool ReaderTryLock() __attribute__((shared_trylock_function(true)));
  void LockWhen(const int &cond) __attribute__((exclusive_lock_function));
};


Mutex sls_mu;

Mutex sls_mu2 __attribute__((acquired_after(sls_mu)));
int sls_guard_var __attribute__((guarded_var)) = 0;
int sls_guardby_var __attribute__((guarded_by(sls_mu))) = 0;

bool getBool();

class MutexWrapper {
public:
   Mutex mu;
   int x __attribute__((guarded_by(mu)));
   void MyLock() __attribute__((exclusive_lock_function(mu))); 
};

MutexWrapper sls_mw;

void sls_fun_0() {
  sls_mw.mu.Lock();
  sls_mw.x = 5; 
  sls_mw.mu.Unlock();
}

void sls_fun_2() {
  sls_mu.Lock();
  int x = sls_guard_var;
  sls_mu.Unlock();
}

void sls_fun_3() {
  sls_mu.Lock();
  sls_guard_var = 2;
  sls_mu.Unlock();
}

void sls_fun_4() {
  sls_mu2.Lock();
  sls_guard_var = 2;
  sls_mu2.Unlock();
}

void sls_fun_5() {
  sls_mu.Lock();
  int x = sls_guardby_var;
  sls_mu.Unlock();
}

void sls_fun_6() {
  sls_mu.Lock();
  sls_guardby_var = 2;
  sls_mu.Unlock();
}

void sls_fun_7() {
  sls_mu.Lock();
  sls_mu2.Lock();
  sls_mu2.Unlock();
  sls_mu.Unlock();
}

void sls_fun_8() {
  sls_mu.Lock();
  if (getBool())
    sls_mu.Unlock();
  else
    sls_mu.Unlock();
}

void sls_fun_9() {
  if (getBool())
    sls_mu.Lock();
  else
    sls_mu.Lock();
  sls_mu.Unlock();
}

void sls_fun_good_6() {
  if (getBool()) {
    sls_mu.Lock();
  } else {
    if (getBool()) {
      getBool(); // EMPTY
    } else {
      getBool(); // EMPTY
    }
    sls_mu.Lock();
  }
  sls_mu.Unlock();
}

void sls_fun_good_7() {
  sls_mu.Lock();
  while (getBool()) {
    sls_mu.Unlock();
    if (getBool()) {
      if (getBool()) {
        sls_mu.Lock();
        continue;
      }
    }
    sls_mu.Lock();
  }
  sls_mu.Unlock();
}

void sls_fun_good_8() {
  sls_mw.MyLock();
  sls_mw.mu.Unlock();
}

void sls_fun_bad_1() {
  sls_mu.Unlock(); // \
    // expected-warning{{unlocking 'sls_mu' that was not locked}}
}

void sls_fun_bad_2() {
  sls_mu.Lock();
  sls_mu.Lock(); // \
    // expected-warning{{locking 'sls_mu' that is already locked}}
  sls_mu.Unlock();
}

void sls_fun_bad_3() {
  sls_mu.Lock(); // \
    // expected-warning{{mutex 'sls_mu' is still locked at the end of function 'sls_fun_bad_3'}}
}

void sls_fun_bad_4() {
  if (getBool())
    sls_mu.Lock(); // \
      // expected-warning{{mutex 'sls_mu' is still locked at the end of its scope}}
  else
    sls_mu2.Lock(); // \
      // expected-warning{{mutex 'sls_mu2' is still locked at the end of its scope}}
}

void sls_fun_bad_5() {
  sls_mu.Lock(); // \
    // expected-warning{{mutex 'sls_mu' is still locked at the end of its scope}}
  if (getBool())
    sls_mu.Unlock();
}

void sls_fun_bad_6() {
  if (getBool()) {
    sls_mu.Lock(); // \
      // expected-warning{{mutex 'sls_mu' is still locked at the end of its scope}}
  } else {
    if (getBool()) {
      getBool(); // EMPTY
    } else {
      getBool(); // EMPTY
    }
  }
  sls_mu.Unlock(); // \
    // expected-warning{{unlocking 'sls_mu' that was not locked}}
}

void sls_fun_bad_7() {
  sls_mu.Lock(); // \
    // expected-warning{{expecting mutex 'sls_mu' to be locked at start of each loop}}
  while (getBool()) {
    sls_mu.Unlock();
    if (getBool()) {
      if (getBool()) {
        continue;
      }
    }
    sls_mu.Lock(); // \
      // expected-warning{{mutex 'sls_mu' is still locked at the end of its scope}}
  }
  sls_mu.Unlock();
}

void sls_fun_bad_8() {
  sls_mu.Lock(); // \
    // expected-warning{{expecting mutex 'sls_mu' to be locked at start of each loop}}
  do {
    sls_mu.Unlock();
  } while (getBool());
}

void sls_fun_bad_9() {
  do {
    sls_mu.Lock(); // \
      // expected-warning{{expecting mutex 'sls_mu' to be locked at start of each loop}}
  } while (getBool());
  sls_mu.Unlock();
}

void sls_fun_bad_10() {
  sls_mu.Lock(); // \
    // expected-warning{{mutex 'sls_mu' is still locked at the end of function 'sls_fun_bad_10'}} \
    // expected-warning{{expecting mutex 'sls_mu' to be locked at start of each loop}}
  while(getBool()) {
    sls_mu.Unlock();
  }
}

void sls_fun_bad_11() {
  while (getBool()) {
    sls_mu.Lock(); // \
      // expected-warning{{expecting mutex 'sls_mu' to be locked at start of each loop}}
  }
  sls_mu.Unlock(); // \
    // expected-warning{{unlocking 'sls_mu' that was not locked}}
}

//-----------------------------------------//
// Handling lock expressions in attribute args
// -------------------------------------------//

Mutex aa_mu;

class GlobalLocker {
public:
  void globalLock() __attribute__((exclusive_lock_function(aa_mu)));
  void globalUnlock() __attribute__((unlock_function(aa_mu)));
};

GlobalLocker glock;

void aa_fun_1() {
  glock.globalLock();
  glock.globalUnlock();
}

void aa_fun_bad_1() {
  glock.globalUnlock(); // \
    // expected-warning{{unlocking 'aa_mu' that was not locked}}
}

void aa_fun_bad_2() {
  glock.globalLock();
  glock.globalLock(); // \
    // expected-warning{{locking 'aa_mu' that is already locked}}
  glock.globalUnlock();
}

void aa_fun_bad_3() {
  glock.globalLock(); // \
    // expected-warning{{mutex 'aa_mu' is still locked at the end of function 'aa_fun_bad_3'}}
}

//--------------------------------------------------//
// Regression tests for unusual method names
//--------------------------------------------------//

Mutex wmu;

// Test diagnostics for other method names.
class WeirdMethods {
  WeirdMethods() {
    wmu.Lock(); // \
      // expected-warning {{mutex 'wmu' is still locked at the end of function 'WeirdMethods'}}
  }
  ~WeirdMethods() {
    wmu.Lock(); // \
      // expected-warning {{mutex 'wmu' is still locked at the end of function '~WeirdMethods'}}
  }
  void operator++() {
    wmu.Lock(); // \
      // expected-warning {{mutex 'wmu' is still locked at the end of function 'operator++'}}
  }
  operator int*() {
    wmu.Lock(); // \
      // expected-warning {{mutex 'wmu' is still locked at the end of function 'operator int *'}}
    return 0;
  }
};

//-----------------------------------------------//
// Errors for guarded by or guarded var variables
// ----------------------------------------------//

int *pgb_gvar __attribute__((pt_guarded_var));
int *pgb_var __attribute__((pt_guarded_by(sls_mu)));

class PGBFoo {
 public:
  int x;
  int *pgb_field __attribute__((guarded_by(sls_mu2)))
                 __attribute__((pt_guarded_by(sls_mu)));
  void testFoo() {
    pgb_field = &x; // \
      // expected-warning {{writing variable 'pgb_field' requires locking 'sls_mu2' exclusively}}
    *pgb_field = x; // expected-warning {{reading variable 'pgb_field' requires locking 'sls_mu2'}} \
      // expected-warning {{writing the value pointed to by 'pgb_field' requires locking 'sls_mu' exclusively}}
    x = *pgb_field; // expected-warning {{reading variable 'pgb_field' requires locking 'sls_mu2'}} \
      // expected-warning {{reading the value pointed to by 'pgb_field' requires locking 'sls_mu'}}
    (*pgb_field)++; // expected-warning {{reading variable 'pgb_field' requires locking 'sls_mu2'}} \
      // expected-warning {{writing the value pointed to by 'pgb_field' requires locking 'sls_mu' exclusively}}
  }
};

class GBFoo {
 public:
  int gb_field __attribute__((guarded_by(sls_mu)));

  void testFoo() {
    gb_field = 0; // \
      // expected-warning {{writing variable 'gb_field' requires locking 'sls_mu' exclusively}}
  }

  void testNoAnal() __attribute__((no_thread_safety_analysis)) {
    gb_field = 0;
  }
};

GBFoo GlobalGBFoo __attribute__((guarded_by(sls_mu)));

void gb_fun_0() {
  sls_mu.Lock();
  int x = *pgb_var;
  sls_mu.Unlock();
}

void gb_fun_1() {
  sls_mu.Lock();
  *pgb_var = 2;
  sls_mu.Unlock();
}

void gb_fun_2() {
  int x;
  pgb_var = &x;
}

void gb_fun_3() {
  int *x = pgb_var;
}

void gb_bad_0() {
  sls_guard_var = 1; // \
    // expected-warning{{writing variable 'sls_guard_var' requires locking any mutex exclusively}}
}

void gb_bad_1() {
  int x = sls_guard_var; // \
    // expected-warning{{reading variable 'sls_guard_var' requires locking any mutex}}
}

void gb_bad_2() {
  sls_guardby_var = 1; // \
    // expected-warning {{writing variable 'sls_guardby_var' requires locking 'sls_mu' exclusively}}
}

void gb_bad_3() {
  int x = sls_guardby_var; // \
    // expected-warning {{reading variable 'sls_guardby_var' requires locking 'sls_mu'}}
}

void gb_bad_4() {
  *pgb_gvar = 1; // \
    // expected-warning {{writing the value pointed to by 'pgb_gvar' requires locking any mutex exclusively}}
}

void gb_bad_5() {
  int x = *pgb_gvar; // \
    // expected-warning {{reading the value pointed to by 'pgb_gvar' requires locking any mutex}}
}

void gb_bad_6() {
  *pgb_var = 1; // \
    // expected-warning {{writing the value pointed to by 'pgb_var' requires locking 'sls_mu' exclusively}}
}

void gb_bad_7() {
  int x = *pgb_var; // \
    // expected-warning {{reading the value pointed to by 'pgb_var' requires locking 'sls_mu'}}
}

void gb_bad_8() {
  GBFoo G;
  G.gb_field = 0; // \
    // expected-warning {{writing variable 'gb_field' requires locking 'sls_mu'}}
}

void gb_bad_9() {
  sls_guard_var++; // \
    // expected-warning{{writing variable 'sls_guard_var' requires locking any mutex exclusively}}
  sls_guard_var--; // \
    // expected-warning{{writing variable 'sls_guard_var' requires locking any mutex exclusively}}
  ++sls_guard_var; // \
    // expected-warning{{writing variable 'sls_guard_var' requires locking any mutex exclusively}}
  --sls_guard_var;// \
    // expected-warning{{writing variable 'sls_guard_var' requires locking any mutex exclusively}}
}

//-----------------------------------------------//
// Warnings on variables with late parsed attributes
// ----------------------------------------------//

class LateFoo {
public:
  int a __attribute__((guarded_by(mu)));
  int b;

  void foo() __attribute__((exclusive_locks_required(mu))) { }

  void test() {
    a = 0; // \
      // expected-warning{{writing variable 'a' requires locking 'mu' exclusively}}
    b = a; // \
      // expected-warning {{reading variable 'a' requires locking 'mu'}}
    c = 0; // \
      // expected-warning {{writing variable 'c' requires locking 'mu' exclusively}}
  }

  int c __attribute__((guarded_by(mu)));

  Mutex mu;
};

class LateBar {
 public:
  int a_ __attribute__((guarded_by(mu1_)));
  int b_;
  int *q __attribute__((pt_guarded_by(mu)));
  Mutex mu1_;
  Mutex mu;
  LateFoo Foo;
  LateFoo Foo2;
  LateFoo *FooPointer;
};

LateBar b1, *b3;

void late_0() {
  LateFoo FooA;
  LateFoo FooB;
  FooA.mu.Lock();
  FooA.a = 5;
  FooA.mu.Unlock();
}

void late_1() {
  LateBar BarA;
  BarA.FooPointer->mu.Lock();
  BarA.FooPointer->a = 2;
  BarA.FooPointer->mu.Unlock();
}

void late_bad_0() {
  LateFoo fooA;
  LateFoo fooB;
  fooA.mu.Lock();
  fooB.a = 5; // \
    // expected-warning{{writing variable 'a' requires locking 'mu' exclusively}}
  fooA.mu.Unlock();
}

void late_bad_1() {
  Mutex mu;
  mu.Lock();
  b1.mu1_.Lock();
  int res = b1.a_ + b3->b_;
  b3->b_ = *b1.q; // \
    // expected-warning{{reading the value pointed to by 'q' requires locking 'mu'}}
  b1.mu1_.Unlock();
  b1.b_ = res;
  mu.Unlock();
}

void late_bad_2() {
  LateBar BarA;
  BarA.FooPointer->mu.Lock();
  BarA.Foo.a = 2; // \
    // expected-warning{{writing variable 'a' requires locking 'mu' exclusively}}
  BarA.FooPointer->mu.Unlock();
}

void late_bad_3() {
  LateBar BarA;
  BarA.Foo.mu.Lock();
  BarA.FooPointer->a = 2; // \
    // expected-warning{{writing variable 'a' requires locking 'mu' exclusively}}
  BarA.Foo.mu.Unlock();
}

void late_bad_4() {
  LateBar BarA;
  BarA.Foo.mu.Lock();
  BarA.Foo2.a = 2; // \
    // expected-warning{{writing variable 'a' requires locking 'mu' exclusively}}
  BarA.Foo.mu.Unlock();
}

//-----------------------------------------------//
// Extra warnings for shared vs. exclusive locks
// ----------------------------------------------//

void shared_fun_0() {
  sls_mu.Lock();
  do {
    sls_mu.Unlock();
    sls_mu.Lock();
  } while (getBool());
  sls_mu.Unlock();
}

void shared_fun_1() {
  sls_mu.ReaderLock(); // \
    // expected-warning {{mutex 'sls_mu' is locked exclusively and shared in the same scope}}
  do {
    sls_mu.Unlock();
    sls_mu.Lock();  // \
      // expected-note {{the other lock of mutex 'sls_mu' is here}}
  } while (getBool());
  sls_mu.Unlock();
}

void shared_fun_3() {
  if (getBool())
    sls_mu.Lock();
  else
    sls_mu.Lock();
  *pgb_var = 1;
  sls_mu.Unlock();
}

void shared_fun_4() {
  if (getBool())
    sls_mu.ReaderLock();
  else
    sls_mu.ReaderLock();
  int x = sls_guardby_var;
  sls_mu.Unlock();
}

void shared_fun_8() {
  if (getBool())
    sls_mu.Lock(); // \
      // expected-warning {{mutex 'sls_mu' is locked exclusively and shared in the same scope}}
  else
    sls_mu.ReaderLock(); // \
      // expected-note {{the other lock of mutex 'sls_mu' is here}}
  sls_mu.Unlock();
}

void shared_bad_0() {
  sls_mu.Lock();  // \
    // expected-warning {{mutex 'sls_mu' is locked exclusively and shared in the same scope}}
  do {
    sls_mu.Unlock();
    sls_mu.ReaderLock();  // \
      // expected-note {{the other lock of mutex 'sls_mu' is here}}
  } while (getBool());
  sls_mu.Unlock();
}

void shared_bad_1() {
  if (getBool())
    sls_mu.Lock(); // \
      // expected-warning {{mutex 'sls_mu' is locked exclusively and shared in the same scope}}
  else
    sls_mu.ReaderLock(); // \
      // expected-note {{the other lock of mutex 'sls_mu' is here}}
  *pgb_var = 1;
  sls_mu.Unlock();
}

void shared_bad_2() {
  if (getBool())
    sls_mu.ReaderLock(); // \
      // expected-warning {{mutex 'sls_mu' is locked exclusively and shared in the same scope}}
  else
    sls_mu.Lock(); // \
      // expected-note {{the other lock of mutex 'sls_mu' is here}}
  *pgb_var = 1;
  sls_mu.Unlock();
}

// FIXME: Add support for functions (not only methods)
class LRBar {
 public:
  void aa_elr_fun() __attribute__((exclusive_locks_required(aa_mu)));
  void aa_elr_fun_s() __attribute__((shared_locks_required(aa_mu)));
  void le_fun() __attribute__((locks_excluded(sls_mu)));
};

class LRFoo {
 public:
  void test() __attribute__((exclusive_locks_required(sls_mu)));
  void testShared() __attribute__((shared_locks_required(sls_mu2)));
};

void elr_fun() __attribute__((exclusive_locks_required(sls_mu)));
void elr_fun() {}

LRFoo MyLRFoo;
LRBar Bar;

void es_fun_0() {
  aa_mu.Lock();
  Bar.aa_elr_fun();
  aa_mu.Unlock();
}

void es_fun_1() {
  aa_mu.Lock();
  Bar.aa_elr_fun_s();
  aa_mu.Unlock();
}

void es_fun_2() {
  aa_mu.ReaderLock();
  Bar.aa_elr_fun_s();
  aa_mu.Unlock();
}

void es_fun_3() {
  sls_mu.Lock();
  MyLRFoo.test();
  sls_mu.Unlock();
}

void es_fun_4() {
  sls_mu2.Lock();
  MyLRFoo.testShared();
  sls_mu2.Unlock();
}

void es_fun_5() {
  sls_mu2.ReaderLock();
  MyLRFoo.testShared();
  sls_mu2.Unlock();
}

void es_fun_6() {
  Bar.le_fun();
}

void es_fun_7() {
  sls_mu.Lock();
  elr_fun();
  sls_mu.Unlock();
}

void es_fun_8() __attribute__((no_thread_safety_analysis));

void es_fun_8() {
  Bar.aa_elr_fun_s();
}

void es_fun_9() __attribute__((shared_locks_required(aa_mu)));
void es_fun_9() {
  Bar.aa_elr_fun_s();
}

void es_fun_10() __attribute__((exclusive_locks_required(aa_mu)));
void es_fun_10() {
  Bar.aa_elr_fun_s();
}

void es_bad_0() {
  Bar.aa_elr_fun(); // \
    // expected-warning {{calling function 'aa_elr_fun' requires exclusive lock on 'aa_mu'}}
}

void es_bad_1() {
  aa_mu.ReaderLock();
  Bar.aa_elr_fun(); // \
    // expected-warning {{calling function 'aa_elr_fun' requires exclusive lock on 'aa_mu'}}
  aa_mu.Unlock();
}

void es_bad_2() {
  Bar.aa_elr_fun_s(); // \
    // expected-warning {{calling function 'aa_elr_fun_s' requires shared lock on 'aa_mu'}}
}

void es_bad_3() {
  MyLRFoo.test(); // \
    // expected-warning {{calling function 'test' requires exclusive lock on 'sls_mu'}}
}

void es_bad_4() {
  MyLRFoo.testShared(); // \
    // expected-warning {{calling function 'testShared' requires shared lock on 'sls_mu2'}}
}

void es_bad_5() {
  sls_mu.ReaderLock();
  MyLRFoo.test(); // \
    // expected-warning {{calling function 'test' requires exclusive lock on 'sls_mu'}}
  sls_mu.Unlock();
}

void es_bad_6() {
  sls_mu.Lock();
  Bar.le_fun(); // \
    // expected-warning {{cannot call function 'le_fun' while mutex 'sls_mu' is locked}}
  sls_mu.Unlock();
}

void es_bad_7() {
  sls_mu.ReaderLock();
  Bar.le_fun(); // \
    // expected-warning {{cannot call function 'le_fun' while mutex 'sls_mu' is locked}}
  sls_mu.Unlock();
}

//-----------------------------------------------//
// Unparseable lock expressions
// ----------------------------------------------//

Mutex UPmu;
// FIXME: add support for lock expressions involving arrays.
Mutex mua[5];

int x __attribute__((guarded_by(UPmu = sls_mu))); // \
  // expected-warning{{cannot resolve lock expression to a specific lockable object}}
int y __attribute__((guarded_by(mua[0]))); // \
  // expected-warning{{cannot resolve lock expression to a specific lockable object}}


void testUnparse() {
  // no errors, since the lock expressions are not resolved
  x = 5;
  y = 5;
}

void testUnparse2() {
  mua[0].Lock(); // \
    // expected-warning{{cannot resolve lock expression to a specific lockable object}}
  (&(mua[0]) + 4)->Lock(); // \
    // expected-warning{{cannot resolve lock expression to a specific lockable object}}
}


//----------------------------------------------------------------------------//
// The following test cases are ported from the gcc thread safety implementation
// They are each wrapped inside a namespace with the test number of the gcc test
//
// FIXME: add all the gcc tests, once this analysis passes them.
//----------------------------------------------------------------------------//

//-----------------------------------------//
// Good testcases (no errors)
//-----------------------------------------//

namespace thread_annot_lock_20 {
class Bar {
 public:
  static int func1() EXCLUSIVE_LOCKS_REQUIRED(mu1_);
  static int b_ GUARDED_BY(mu1_);
  static Mutex mu1_;
  static int a_ GUARDED_BY(mu1_);
};

Bar b1;

int Bar::func1()
{
  int res = 5;

  if (a_ == 4)
    res = b_;
  return res;
}
} // end namespace thread_annot_lock_20

namespace thread_annot_lock_22 {
// Test various usage of GUARDED_BY and PT_GUARDED_BY annotations, especially
// uses in class definitions.
Mutex mu;

class Bar {
 public:
  int a_ GUARDED_BY(mu1_);
  int b_;
  int *q PT_GUARDED_BY(mu);
  Mutex mu1_ ACQUIRED_AFTER(mu);
};

Bar b1, *b3;
int *p GUARDED_BY(mu) PT_GUARDED_BY(mu);
int res GUARDED_BY(mu) = 5;

int func(int i)
{
  int x;
  mu.Lock();
  b1.mu1_.Lock();
  res = b1.a_ + b3->b_;
  *p = i;
  b1.a_ = res + b3->b_;
  b3->b_ = *b1.q;
  b1.mu1_.Unlock();
  b1.b_ = res;
  x = res;
  mu.Unlock();
  return x;
}
} // end namespace thread_annot_lock_22

namespace thread_annot_lock_27_modified {
// test lock annotations applied to function definitions
// Modified: applied annotations only to function declarations
Mutex mu1;
Mutex mu2 ACQUIRED_AFTER(mu1);

class Foo {
 public:
  int method1(int i) SHARED_LOCKS_REQUIRED(mu2) EXCLUSIVE_LOCKS_REQUIRED(mu1);
};

int Foo::method1(int i) {
  return i;
}


int foo(int i) EXCLUSIVE_LOCKS_REQUIRED(mu2) SHARED_LOCKS_REQUIRED(mu1);
int foo(int i) {
  return i;
}

static int bar(int i) EXCLUSIVE_LOCKS_REQUIRED(mu1);
static int bar(int i) {
  return i;
}

void main() {
  Foo a;

  mu1.Lock();
  mu2.Lock();
  a.method1(1);
  foo(2);
  mu2.Unlock();
  bar(3);
  mu1.Unlock();
}
} // end namespace thread_annot_lock_27_modified


namespace thread_annot_lock_38 {
// Test the case where a template member function is annotated with lock
// attributes in a non-template class.
class Foo {
 public:
  void func1(int y) LOCKS_EXCLUDED(mu_);
  template <typename T> void func2(T x) LOCKS_EXCLUDED(mu_);
 private:
  Mutex mu_;
};

Foo *foo;

void main()
{
  foo->func1(5);
  foo->func2(5);
}
} // end namespace thread_annot_lock_38

namespace thread_annot_lock_43 {
// Tests lock canonicalization
class Foo {
 public:
  Mutex *mu_;
};

class FooBar {
 public:
  Foo *foo_;
  int GetA() EXCLUSIVE_LOCKS_REQUIRED(foo_->mu_) { return a_; }
  int a_ GUARDED_BY(foo_->mu_);
};

FooBar *fb;

void main()
{
  int x;
  fb->foo_->mu_->Lock();
  x = fb->GetA();
  fb->foo_->mu_->Unlock();
}
} // end namespace thread_annot_lock_43

namespace thread_annot_lock_49 {
// Test the support for use of lock expression in the annotations
class Foo {
 public:
  Mutex foo_mu_;
};

class Bar {
 private:
  Foo *foo;
  Mutex bar_mu_ ACQUIRED_AFTER(foo->foo_mu_);

 public:
  void Test1() {
    foo->foo_mu_.Lock();
    bar_mu_.Lock();
    bar_mu_.Unlock();
    foo->foo_mu_.Unlock();
  }
};

void main() {
  Bar bar;
  bar.Test1();
}
} // end namespace thread_annot_lock_49

namespace thread_annot_lock_61_modified {
  // Modified to fix the compiler errors
  // Test the fix for a bug introduced by the support of pass-by-reference
  // paramters.
  struct Foo { Foo &operator<< (bool) {return *this;} };
  Foo &getFoo();
  struct Bar { Foo &func () {return getFoo();} };
  struct Bas { void operator& (Foo &) {} };
  void mumble()
  {
    Bas() & Bar().func() << "" << "";
    Bas() & Bar().func() << "";
  }
} // end namespace thread_annot_lock_61_modified


namespace thread_annot_lock_65 {
// Test the fix for a bug in the support of allowing reader locks for
// non-const, non-modifying overload functions. (We didn't handle the builtin
// properly.)
enum MyFlags {
  Zero,
  One,
  Two,
  Three,
  Four,
  Five,
  Six,
  Seven,
  Eight,
  Nine
};

inline MyFlags
operator|(MyFlags a, MyFlags b)
{
  return MyFlags(static_cast<int>(a) | static_cast<int>(b));
}

inline MyFlags&
operator|=(MyFlags& a, MyFlags b)
{
    return a = a | b;
}
} // end namespace thread_annot_lock_65

namespace thread_annot_lock_66_modified {
// Modified: Moved annotation to function defn
// Test annotations on out-of-line definitions of member functions where the
// annotations refer to locks that are also data members in the class.
Mutex mu;

class Foo {
 public:
  int method1(int i) SHARED_LOCKS_REQUIRED(mu1, mu, mu2);
  int data GUARDED_BY(mu1);
  Mutex *mu1;
  Mutex *mu2;
};

int Foo::method1(int i)
{
  return data + i;
}

void main()
{
  Foo a;

  a.mu2->Lock();
  a.mu1->Lock();
  mu.Lock();
  a.method1(1);
  mu.Unlock();
  a.mu1->Unlock();
  a.mu2->Unlock();
}
} // end namespace thread_annot_lock_66_modified

namespace thread_annot_lock_68_modified {
// Test a fix to a bug in the delayed name binding with nested template
// instantiation. We use a stack to make sure a name is not resolved to an
// inner context.
template <typename T>
class Bar {
  Mutex mu_;
};

template <typename T>
class Foo {
 public:
  void func(T x) {
    mu_.Lock();
    count_ = x;
    mu_.Unlock();
  }

 private:
  T count_ GUARDED_BY(mu_);
  Bar<T> bar_;
  Mutex mu_;
};

void main()
{
  Foo<int> *foo;
  foo->func(5);
}
} // end namespace thread_annot_lock_68_modified

namespace thread_annot_lock_30_modified {
// Test delay parsing of lock attribute arguments with nested classes.
// Modified: trylocks replaced with exclusive_lock_fun
int a = 0;

class Bar {
  struct Foo;

 public:
  void MyLock() EXCLUSIVE_LOCK_FUNCTION(mu);

  int func() {
    MyLock();
//    if (foo == 0) {
//      return 0;
//    }
    a = 5;
    mu.Unlock();
    return 1;
  }

  class FooBar {
    int x;
    int y;
  };

 private:
  Mutex mu;
};

Bar *bar;

void main()
{
  bar->func();
}
} // end namespace thread_annot_lock_30_modified

namespace thread_annot_lock_47 {
// Test the support for annotations on virtual functions.
// This is a good test case. (i.e. There should be no warning emitted by the
// compiler.)
class Base {
 public:
  virtual void func1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual void func2() LOCKS_EXCLUDED(mu_);
  Mutex mu_;
};

class Child : public Base {
 public:
  virtual void func1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual void func2() LOCKS_EXCLUDED(mu_);
};

void main() {
  Child *c;
  Base *b = c;

  b->mu_.Lock();
  b->func1();
  b->mu_.Unlock();
  b->func2();

  c->mu_.Lock();
  c->func1();
  c->mu_.Unlock();
  c->func2();
}
} // end namespace thread_annot_lock_47

//-----------------------------------------//
// Tests which produce errors
//-----------------------------------------//

namespace thread_annot_lock_13 {
Mutex mu1;
Mutex mu2;

int g GUARDED_BY(mu1);
int w GUARDED_BY(mu2);

class Foo {
 public:
  void bar() LOCKS_EXCLUDED(mu_, mu1);
  int foo() SHARED_LOCKS_REQUIRED(mu_) EXCLUSIVE_LOCKS_REQUIRED(mu2);

 private:
  int a_ GUARDED_BY(mu_);
 public:
  Mutex mu_ ACQUIRED_AFTER(mu1);
};

int Foo::foo()
{
  int res;
  w = 5.2;
  res = a_ + 5;
  return res;
}

void Foo::bar()
{
  int x;
  mu_.Lock();
  x = foo(); // expected-warning {{calling function 'foo' requires exclusive lock on 'mu2'}}
  a_ = x + 1;
  mu_.Unlock();
  if (x > 5) {
    mu1.Lock();
    g = 2.3;
    mu1.Unlock();
  }
}

void main()
{
  Foo f1, *f2;
  f1.mu_.Lock();
  f1.bar(); // expected-warning {{cannot call function 'bar' while mutex 'mu_' is locked}}
  mu2.Lock();
  f1.foo();
  mu2.Unlock();
  f1.mu_.Unlock();
  f2->mu_.Lock();
  f2->bar(); // expected-warning {{cannot call function 'bar' while mutex 'mu_' is locked}}
  f2->mu_.Unlock();
  mu2.Lock();
  w = 2.5;
  mu2.Unlock();
}
} // end namespace thread_annot_lock_13

namespace thread_annot_lock_18_modified {
// Modified: Trylocks removed
// Test the ability to distnguish between the same lock field of
// different objects of a class.
  class Bar {
 public:
  bool MyLock() EXCLUSIVE_LOCK_FUNCTION(mu1_);
  void MyUnlock() UNLOCK_FUNCTION(mu1_);
  int a_ GUARDED_BY(mu1_);

 private:
  Mutex mu1_;
};

Bar *b1, *b2;

void func()
{
  b1->MyLock();
  b1->a_ = 5;
  b2->a_ = 3; // expected-warning {{writing variable 'a_' requires locking 'mu1_' exclusively}}
  b2->MyLock();
  b2->MyUnlock();
  b1->MyUnlock();
}
} // end namespace thread_annot_lock_18_modified

namespace thread_annot_lock_21 {
// Test various usage of GUARDED_BY and PT_GUARDED_BY annotations, especially
// uses in class definitions.
Mutex mu;

class Bar {
 public:
  int a_ GUARDED_BY(mu1_);
  int b_;
  int *q PT_GUARDED_BY(mu);
  Mutex mu1_ ACQUIRED_AFTER(mu);
};

Bar b1, *b3;
int *p GUARDED_BY(mu) PT_GUARDED_BY(mu);

int res GUARDED_BY(mu) = 5;

int func(int i)
{
  int x;
  b3->mu1_.Lock();
  res = b1.a_ + b3->b_; // expected-warning {{reading variable 'a_' requires locking 'mu1_'}} \
    // expected-warning {{writing variable 'res' requires locking 'mu' exclusively}}
  *p = i; // expected-warning {{reading variable 'p' requires locking 'mu'}} \
    // expected-warning {{writing the value pointed to by 'p' requires locking 'mu' exclusively}}
  b1.a_ = res + b3->b_; // expected-warning {{reading variable 'res' requires locking 'mu'}} \
    // expected-warning {{writing variable 'a_' requires locking 'mu1_' exclusively}}
  b3->b_ = *b1.q; // expected-warning {{reading the value pointed to by 'q' requires locking 'mu'}}
  b3->mu1_.Unlock();
  b1.b_ = res; // expected-warning {{reading variable 'res' requires locking 'mu'}}
  x = res; // expected-warning {{reading variable 'res' requires locking 'mu'}}
  return x;
}
} // end namespace thread_annot_lock_21

namespace thread_annot_lock_35_modified {
// Test the analyzer's ability to distinguish the lock field of different
// objects.
class Foo {
 private:
  Mutex lock_;
  int a_ GUARDED_BY(lock_);

 public:
  void Func(Foo* child) LOCKS_EXCLUDED(lock_) {
     Foo *new_foo = new Foo;

     lock_.Lock();

     child->Func(new_foo); // There shouldn't be any warning here as the
                           // acquired lock is not in child.
     child->bar(7); // expected-warning {{calling function 'bar' requires exclusive lock on 'lock_'}}
     child->a_ = 5; // expected-warning {{writing variable 'a_' requires locking 'lock_' exclusively}}
     lock_.Unlock();
  }

  void bar(int y) EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    a_ = y;
  }
};

Foo *x;

void main() {
  Foo *child = new Foo;
  x->Func(child);
}
} // end namespace thread_annot_lock_35_modified

namespace thread_annot_lock_36_modified {
// Modified to move the annotations to function defns.
// Test the analyzer's ability to distinguish the lock field of different
// objects
class Foo {
 private:
  Mutex lock_;
  int a_ GUARDED_BY(lock_);

 public:
  void Func(Foo* child) LOCKS_EXCLUDED(lock_);
  void bar(int y) EXCLUSIVE_LOCKS_REQUIRED(lock_);
};

void Foo::Func(Foo* child) {
  Foo *new_foo = new Foo;

  lock_.Lock();

  child->lock_.Lock();
  child->Func(new_foo); // expected-warning {{cannot call function 'Func' while mutex 'lock_' is locked}}
  child->bar(7);
  child->a_ = 5;
  child->lock_.Unlock();

  lock_.Unlock();
}

void Foo::bar(int y) {
  a_ = y;
}


Foo *x;

void main() {
  Foo *child = new Foo;
  x->Func(child);
}
} // end namespace thread_annot_lock_36_modified


namespace thread_annot_lock_42 {
// Test support of multiple lock attributes of the same kind on a decl.
class Foo {
 private:
  Mutex mu1, mu2, mu3;
  int x GUARDED_BY(mu1) GUARDED_BY(mu2);
  int y GUARDED_BY(mu2);

  void f2() LOCKS_EXCLUDED(mu1) LOCKS_EXCLUDED(mu2) LOCKS_EXCLUDED(mu3) {
    mu2.Lock();
    y = 2;
    mu2.Unlock();
  }

 public:
  void f1() EXCLUSIVE_LOCKS_REQUIRED(mu2) EXCLUSIVE_LOCKS_REQUIRED(mu1) {
    x = 5;
    f2(); // expected-warning {{cannot call function 'f2' while mutex 'mu1' is locked}} \
      // expected-warning {{cannot call function 'f2' while mutex 'mu2' is locked}}
  }
};

Foo *foo;

void func()
{
  foo->f1(); // expected-warning {{calling function 'f1' requires exclusive lock on 'mu2'}} \
    // expected-warning {{calling function 'f1' requires exclusive lock on 'mu1'}}
}
} // end namespace thread_annot_lock_42

namespace thread_annot_lock_46 {
// Test the support for annotations on virtual functions.
class Base {
 public:
  virtual void func1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual void func2() LOCKS_EXCLUDED(mu_);
  Mutex mu_;
};

class Child : public Base {
 public:
  virtual void func1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual void func2() LOCKS_EXCLUDED(mu_);
};

void main() {
  Child *c;
  Base *b = c;

  b->func1(); // expected-warning {{calling function 'func1' requires exclusive lock on 'mu_'}}
  b->mu_.Lock();
  b->func2(); // expected-warning {{cannot call function 'func2' while mutex 'mu_' is locked}}
  b->mu_.Unlock();

  c->func1(); // expected-warning {{calling function 'func1' requires exclusive lock on 'mu_'}}
  c->mu_.Lock();
  c->func2(); // expected-warning {{cannot call function 'func2' while mutex 'mu_' is locked}}
  c->mu_.Unlock();
}
} // end namespace thread_annot_lock_46

namespace thread_annot_lock_67_modified {
// Modified: attributes on definitions moved to declarations
// Test annotations on out-of-line definitions of member functions where the
// annotations refer to locks that are also data members in the class.
Mutex mu;
Mutex mu3;

class Foo {
 public:
  int method1(int i) SHARED_LOCKS_REQUIRED(mu1, mu, mu2, mu3);
  int data GUARDED_BY(mu1);
  Mutex *mu1;
  Mutex *mu2;
};

int Foo::method1(int i) {
  return data + i;
}

void main()
{
  Foo a;
  a.method1(1); // expected-warning {{calling function 'method1' requires shared lock on 'mu1'}} \
    // expected-warning {{calling function 'method1' requires shared lock on 'mu'}} \
    // expected-warning {{calling function 'method1' requires shared lock on 'mu2'}} \
    // expected-warning {{calling function 'method1' requires shared lock on 'mu3'}}
}
} // end namespace thread_annot_lock_67_modified


