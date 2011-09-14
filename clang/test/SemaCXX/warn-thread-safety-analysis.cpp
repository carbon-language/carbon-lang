// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety %s


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
    // expected-warning{{mutex 'sls_mu' is still held at the end of function 'sls_fun_bad_3'}}
}

void sls_fun_bad_4() {
  if (getBool())
    sls_mu.Lock(); // \
      // expected-warning{{mutex 'sls_mu' is still held at the end of its scope}}
  else
    sls_mu2.Lock(); // \
      // expected-warning{{mutex 'sls_mu2' is still held at the end of its scope}}
}

void sls_fun_bad_5() {
  sls_mu.Lock(); // \
    // expected-warning{{mutex 'sls_mu' is still held at the end of its scope}}
  if (getBool())
    sls_mu.Unlock();
}

void sls_fun_bad_6() {
  if (getBool()) {
    sls_mu.Lock(); // \
      // expected-warning{{mutex 'sls_mu' is still held at the end of its scope}}
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
  sls_mu.Lock();
  while (getBool()) { // \
      // expected-warning{{expecting lock on 'sls_mu' to be held at start of each loop}}
    sls_mu.Unlock();
    if (getBool()) {
      if (getBool()) {
        continue;
      }
    }
    sls_mu.Lock(); // \
      // expected-warning{{mutex 'sls_mu' is still held at the end of its scope}}
  }
  sls_mu.Unlock();
}

void sls_fun_bad_8() {
  sls_mu.Lock();
  do {
    sls_mu.Unlock();  // \
      // expected-warning{{expecting lock on 'sls_mu' to be held at start of each loop}}
  } while (getBool());
}

void sls_fun_bad_9() {
  do {
    sls_mu.Lock(); // \
      // expected-warning{{mutex 'sls_mu' is still held at the end of its scope}}
  } while (getBool());
  sls_mu.Unlock();
}

void sls_fun_bad_10() {
  sls_mu.Lock(); // \
    // expected-warning{{mutex 'sls_mu' is still held at the end of function 'sls_fun_bad_10'}}
  while(getBool()) { // \
      // expected-warning{{expecting lock on 'sls_mu' to be held at start of each loop}}
    sls_mu.Unlock();
  }
}

void sls_fun_bad_11() {
  while (getBool()) {
    sls_mu.Lock(); // \
      // expected-warning{{mutex 'sls_mu' is still held at the end of its scope}}
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
    // expected-warning{{mutex 'aa_mu' is still held at the end of function 'aa_fun_bad_3'}}
}

//--------------------------------------------------//
// Regression tests for unusual method names
//--------------------------------------------------//

Mutex wmu;

// Test diagnostics for other method names.
class WeirdMethods {
  WeirdMethods() {
    wmu.Lock(); // \
      // expected-warning {{mutex 'wmu' is still held at the end of function 'WeirdMethods'}}
  }
  ~WeirdMethods() {
    wmu.Lock(); // \
      // expected-warning {{mutex 'wmu' is still held at the end of function '~WeirdMethods'}}
  }
  void operator++() {
    wmu.Lock(); // \
      // expected-warning {{mutex 'wmu' is still held at the end of function 'operator++'}}
  }
  operator int*() {
    wmu.Lock(); // \
      // expected-warning {{mutex 'wmu' is still held at the end of function 'operator int *'}}
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
      // expected-warning {{writing variable 'pgb_field' requires lock on 'sls_mu2' to be held exclusively}}
    *pgb_field = x; // expected-warning {{reading variable 'pgb_field' requires lock on 'sls_mu2' to be held}} \
      // expected-warning {{writing the value pointed to by 'pgb_field' requires lock on 'sls_mu' to be held exclusively}}
    x = *pgb_field; // expected-warning {{reading variable 'pgb_field' requires lock on 'sls_mu2' to be held}} \
      // expected-warning {{reading the value pointed to by 'pgb_field' requires lock on 'sls_mu' to be held}}
    (*pgb_field)++; // expected-warning {{reading variable 'pgb_field' requires lock on 'sls_mu2' to be held}} \
      // expected-warning {{writing the value pointed to by 'pgb_field' requires lock on 'sls_mu' to be held exclusively}}
  }
};

class GBFoo {
 public:
  int gb_field __attribute__((guarded_by(sls_mu)));

  void testFoo() {
    gb_field = 0; // \
      // expected-warning {{writing variable 'gb_field' requires lock on 'sls_mu' to be held exclusively}}
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
    // expected-warning{{writing variable 'sls_guard_var' requires lock on any mutex to be held exclusively}}
}

void gb_bad_1() {
  int x = sls_guard_var; // \
    // expected-warning{{reading variable 'sls_guard_var' requires lock on any mutex to be held}}
}

void gb_bad_2() {
  sls_guardby_var = 1; // \
    // expected-warning {{writing variable 'sls_guardby_var' requires lock on 'sls_mu' to be held exclusively}}
}

void gb_bad_3() {
  int x = sls_guardby_var; // \
    // expected-warning {{reading variable 'sls_guardby_var' requires lock on 'sls_mu' to be held}}
}

void gb_bad_4() {
  *pgb_gvar = 1; // \
    // expected-warning {{writing the value pointed to by 'pgb_gvar' requires lock on any mutex to be held exclusively}}
}

void gb_bad_5() {
  int x = *pgb_gvar; // \
    // expected-warning {{reading the value pointed to by 'pgb_gvar' requires lock on any mutex to be held}}
}

void gb_bad_6() {
  *pgb_var = 1; // \
    // expected-warning {{writing the value pointed to by 'pgb_var' requires lock on 'sls_mu' to be held exclusively}}
}

void gb_bad_7() {
  int x = *pgb_var; // \
    // expected-warning {{reading the value pointed to by 'pgb_var' requires lock on 'sls_mu' to be held}}
}

void gb_bad_8() {
  GBFoo G;
  G.gb_field = 0; // \
    // expected-warning {{writing variable 'gb_field' requires lock on 'sls_mu'}}
}

void gb_bad_9() {
  sls_guard_var++; // \
    // expected-warning{{writing variable 'sls_guard_var' requires lock on any mutex to be held exclusively}}
  sls_guard_var--; // \
    // expected-warning{{writing variable 'sls_guard_var' requires lock on any mutex to be held exclusively}}
  ++sls_guard_var; // \
    // expected-warning{{writing variable 'sls_guard_var' requires lock on any mutex to be held exclusively}}
  --sls_guard_var;// \
    // expected-warning{{writing variable 'sls_guard_var' requires lock on any mutex to be held exclusively}}
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
      // expected-warning{{writing variable 'a' requires lock on 'mu' to be held exclusively}}
    b = a; // \
      // expected-warning {{reading variable 'a' requires lock on 'mu' to be held}}
    c = 0; // \
      // expected-warning {{writing variable 'c' requires lock on 'mu' to be held exclusively}}
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
    // expected-warning{{writing variable 'a' requires lock on 'mu' to be held exclusively}}
  fooA.mu.Unlock();
}

void late_bad_1() {
  Mutex mu;
  mu.Lock();
  b1.mu1_.Lock();
  int res = b1.a_ + b3->b_;
  b3->b_ = *b1.q; // \
    // expected-warning{{reading the value pointed to by 'q' requires lock on 'mu' to be held}}
  b1.mu1_.Unlock();
  b1.b_ = res;
  mu.Unlock();
}

void late_bad_2() {
  LateBar BarA;
  BarA.FooPointer->mu.Lock();
  BarA.Foo.a = 2; // \
    // expected-warning{{writing variable 'a' requires lock on 'mu' to be held exclusively}}
  BarA.FooPointer->mu.Unlock();
}

void late_bad_3() {
  LateBar BarA;
  BarA.Foo.mu.Lock();
  BarA.FooPointer->a = 2; // \
    // expected-warning{{writing variable 'a' requires lock on 'mu' to be held exclusively}}
  BarA.Foo.mu.Unlock();
}

void late_bad_4() {
  LateBar BarA;
  BarA.Foo.mu.Lock();
  BarA.Foo2.a = 2; // \
    // expected-warning{{writing variable 'a' requires lock on 'mu' to be held exclusively}}
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
  sls_mu.ReaderLock();
  do {
    sls_mu.Unlock();
    sls_mu.Lock(); // \
      // expected-warning {{lock 'sls_mu' is exclusive and shared in the same scope}}
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
      // expected-warning {{lock 'sls_mu' is exclusive and shared in the same scope}}
  else
    sls_mu.ReaderLock(); // \
      // expected-note {{the other lock of mutex 'sls_mu' is here}}
  sls_mu.Unlock();
}

void shared_bad_0() {
  sls_mu.Lock();
  do {
    sls_mu.Unlock();
    sls_mu.ReaderLock(); // \
      // expected-warning {{lock 'sls_mu' is exclusive and shared in the same scope}}
  } while (getBool());
  sls_mu.Unlock();
}

void shared_bad_1() {
  if (getBool())
    sls_mu.Lock(); // \
      // expected-warning {{lock 'sls_mu' is exclusive and shared in the same scope}}
  else
    sls_mu.ReaderLock(); // \
      // expected-note {{the other lock of mutex 'sls_mu' is here}}
  *pgb_var = 1;
  sls_mu.Unlock();
}

void shared_bad_2() {
  if (getBool())
    sls_mu.ReaderLock(); // \
      // expected-warning {{lock 'sls_mu' is exclusive and shared in the same scope}}
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
    // expected-warning {{cannot call function 'le_fun' while holding mutex 'sls_mu'}}
  sls_mu.Unlock();
}

void es_bad_7() {
  sls_mu.ReaderLock();
  Bar.le_fun(); // \
    // expected-warning {{cannot call function 'le_fun' while holding mutex 'sls_mu'}}
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

