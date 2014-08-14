// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety -Wthread-safety-beta -Wthread-safety-verbose -Wno-thread-safety-negative -fcxx-exceptions %s

#define LOCKABLE            __attribute__ ((lockable))
#define SCOPED_LOCKABLE     __attribute__ ((scoped_lockable))
#define GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
#define GUARDED_VAR         __attribute__ ((guarded_var))
#define PT_GUARDED_BY(x)    __attribute__ ((pt_guarded_by(x)))
#define PT_GUARDED_VAR      __attribute__ ((pt_guarded_var))
#define ACQUIRED_AFTER(...) __attribute__ ((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...) __attribute__ ((acquired_before(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)    __attribute__ ((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)       __attribute__ ((shared_lock_function(__VA_ARGS__)))
#define ASSERT_EXCLUSIVE_LOCK(...)      __attribute__ ((assert_exclusive_lock(__VA_ARGS__)))
#define ASSERT_SHARED_LOCK(...)         __attribute__ ((assert_shared_lock(__VA_ARGS__)))
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


class  __attribute__((lockable)) Mutex {
 public:
  void Lock() __attribute__((exclusive_lock_function));
  void ReaderLock() __attribute__((shared_lock_function));
  void Unlock() __attribute__((unlock_function));
  bool TryLock() __attribute__((exclusive_trylock_function(true)));
  bool ReaderTryLock() __attribute__((shared_trylock_function(true)));
  void LockWhen(const int &cond) __attribute__((exclusive_lock_function));

  // for negative capabilities
  const Mutex& operator!() const { return *this; }

  void AssertHeld()       ASSERT_EXCLUSIVE_LOCK();
  void AssertReaderHeld() ASSERT_SHARED_LOCK();
};


class Test {
  Mutex mu;
  int a GUARDED_BY(mu);  // expected-note3 {{Guarded_by declared here.}}

  void foo1() EXCLUSIVE_LOCKS_REQUIRED(mu);
  void foo2() SHARED_LOCKS_REQUIRED(mu);
  void foo3() LOCKS_EXCLUDED(mu);

  void test1() {  // expected-note {{Thread warning in function 'test1'}}
    a = 0;        // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
  }

  void test2() {  // expected-note {{Thread warning in function 'test2'}}
    int b = a;    // expected-warning {{reading variable 'a' requires holding mutex 'mu'}}
  }

  void test3() {  // expected-note {{Thread warning in function 'test3'}}
    foo1();       // expected-warning {{calling function 'foo1' requires holding mutex 'mu' exclusively}}
  }

  void test4() {  // expected-note {{Thread warning in function 'test4'}}
    foo2();       // expected-warning {{calling function 'foo2' requires holding mutex 'mu'}}
  }

  void test5() {  // expected-note {{Thread warning in function 'test5'}}
    mu.ReaderLock();
    foo1();       // expected-warning {{calling function 'foo1' requires holding mutex 'mu' exclusively}}
    mu.Unlock();
  }

  void test6() {  // expected-note {{Thread warning in function 'test6'}}
    mu.ReaderLock();
    a = 0;        // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
    mu.Unlock();
  }

  void test7() {  // expected-note {{Thread warning in function 'test7'}}
    mu.Lock();
    foo3();       // expected-warning {{cannot call function 'foo3' while mutex 'mu' is held}}
    mu.Unlock();
  }
};

