// RUN: %clangxx_tsan %s -o %t
// RUN: TSAN_OPTIONS=detect_deadlocks=1 not %t 2>&1 | FileCheck %s
#include <pthread.h>
#undef NDEBUG
#include <assert.h>
#include <stdio.h>

class PthreadMutex {
 public:
  PthreadMutex() { assert(0 == pthread_mutex_init(&mu_, 0)); }
  ~PthreadMutex() {
    assert(0 == pthread_mutex_destroy(&mu_));
    (void)padding_;
  }
  void lock() { assert(0 == pthread_mutex_lock(&mu_)); }
  void unlock() { assert(0 == pthread_mutex_unlock(&mu_)); }
  bool try_lock() { return 0 == pthread_mutex_trylock(&mu_); }

 private:
  pthread_mutex_t mu_;
  char padding_[64 - sizeof(pthread_mutex_t)];
};

class PthreadSpinLock {
 public:
  PthreadSpinLock() { assert(0 == pthread_spin_init(&mu_, 0)); }
  ~PthreadSpinLock() {
    assert(0 == pthread_spin_destroy(&mu_));
    (void)padding_;
  }
  void lock() { assert(0 == pthread_spin_lock(&mu_)); }
  void unlock() { assert(0 == pthread_spin_unlock(&mu_)); }
  bool try_lock() { return 0 == pthread_spin_trylock(&mu_); }

 private:
  pthread_spinlock_t mu_;
  char padding_[64 - sizeof(pthread_spinlock_t)];
};


template <class LockType>
class LockTest {
 public:
  LockTest(size_t n) : n_(n), locks_(new LockType[n]) { }
  ~LockTest() { delete [] locks_; }
  void L(size_t i) {
    assert(i < n_);
    locks_[i].lock();
  }
  void U(size_t i) {
    assert(i < n_);
    locks_[i].unlock();
  }

  void *A(size_t i) {
    assert(i < n_);
    return &locks_[i];
  }

  bool T(size_t i) {
    assert(i < n_);
    return locks_[i].try_lock();
  }

  // Simple lock order onversion.
  void Test1() {
    fprintf(stderr, "Starting Test1\n");
    // CHECK: Starting Test1
    fprintf(stderr, "Expecting lock inversion: %p %p\n", A(0), A(1));
    // CHECK: Expecting lock inversion: [[A1:0x[a-f0-9]*]] [[A2:0x[a-f0-9]*]]
    Lock_0_1();
    Lock_1_0();
    // CHECK: WARNING: ThreadSanitizer: lock-order-inversion (potential deadlock)
    // CHECK: path: [[M1:M[0-9]+]] => [[M2:M[0-9]+]] => [[M1]]
    // CHECK: Mutex [[M1]] ([[A1]]) created at:
    // CHECK: Mutex [[M2]] ([[A2]]) created at:
    // CHECK-NOT: WARNING: ThreadSanitizer:
  }

  // Simple lock order inversion with 3 locks.
  void Test2() {
    fprintf(stderr, "Starting Test2\n");
    // CHECK: Starting Test2
    fprintf(stderr, "Expecting lock inversion: %p %p %p\n", A(0), A(1), A(2));
    // CHECK: Expecting lock inversion: [[A1:0x[a-f0-9]*]] [[A2:0x[a-f0-9]*]] [[A3:0x[a-f0-9]*]]
    Lock2(0, 1);
    Lock2(1, 2);
    Lock2(2, 0);
    // CHECK: WARNING: ThreadSanitizer: lock-order-inversion (potential deadlock)
    // CHECK: path: [[M1:M[0-9]+]] => [[M2:M[0-9]+]] => [[M3:M[0-9]+]] => [[M1]]
    // CHECK: Mutex [[M1]] ([[A1]]) created at:
    // CHECK: Mutex [[M2]] ([[A2]]) created at:
    // CHECK: Mutex [[M3]] ([[A3]]) created at:
    // CHECK-NOT: WARNING: ThreadSanitizer:
  }

  // Lock order inversion with lots of new locks created (but not used)
  // between. Since the new locks are not used we should still detect the
  // deadlock.
  void Test3() {
    fprintf(stderr, "Starting Test3\n");
    // CHECK: Starting Test3
    Lock_0_1();
    L(2);
    CreateAndDestroyManyLocks();
    U(2);
    Lock_1_0();
    // CHECK: WARNING: ThreadSanitizer: lock-order-inversion (potential deadlock)
    // CHECK-NOT: WARNING: ThreadSanitizer:
  }

  // lock l0=>l1; then create and use lots of locks; then lock l1=>l0.
  // The deadlock epoch should have changed and we should not report anything.
  void Test4() {
    fprintf(stderr, "Starting Test4\n");
    // CHECK: Starting Test4
    Lock_0_1();
    L(2);
    CreateLockUnlockAndDestroyManyLocks();
    U(2);
    Lock_1_0();
    // CHECK-NOT: WARNING: ThreadSanitizer:
  }

  void Test5() {
    fprintf(stderr, "Starting Test5\n");
    // CHECK: Starting Test5
    RunThreads(&LockTest::Lock_0_1, &LockTest::Lock_1_0);
    // CHECK: WARNING: ThreadSanitizer: lock-order-inversion
    // CHECK-NOT: WARNING: ThreadSanitizer:
  }

  void Test6() {
    fprintf(stderr, "Starting Test6\n");
    // CHECK: Starting Test6
    // CHECK-NOT: WARNING: ThreadSanitizer:
    RunThreads(&LockTest::Lock1_Loop_0, &LockTest::Lock1_Loop_1,
               &LockTest::Lock1_Loop_2);
  }

  void Test7() {
    fprintf(stderr, "Starting Test7\n");
    // CHECK: Starting Test7
    L(0); T(1); U(1); U(0);
    T(1); L(0); U(1); U(0);
    // CHECK-NOT: WARNING: ThreadSanitizer:
    fprintf(stderr, "No cycle: 0=>1\n");
    // CHECK: No cycle: 0=>1

    T(2); L(3); U(3); U(2);
    L(3); T(2); U(3); U(2);
    // CHECK-NOT: WARNING: ThreadSanitizer:
    fprintf(stderr, "No cycle: 2=>3\n");
    // CHECK: No cycle: 2=>3

    T(4); L(5); U(4); U(5);
    L(5); L(4); U(4); U(5);
    // CHECK: WARNING: ThreadSanitizer: lock-order-inversion
    fprintf(stderr, "Have cycle: 4=>5\n");
    // CHECK: Have cycle: 4=>5

    L(7); L(6); U(6); U(7);
    T(6); L(7); U(6); U(7);
    // CHECK: WARNING: ThreadSanitizer: lock-order-inversion
    fprintf(stderr, "Have cycle: 6=>7\n");
    // CHECK: Have cycle: 6=>7
  }

 private:
  void Lock2(size_t l1, size_t l2) { L(l1); L(l2); U(l2); U(l1); }
  void Lock_0_1() { Lock2(0, 1); }
  void Lock_1_0() { Lock2(1, 0); }
  void Lock1_Loop(size_t i, size_t n_iter) {
    for (size_t it = 0; it < n_iter; it++) {
      // if ((it & (it - 1)) == 0) fprintf(stderr, "%zd", i);
      L(i);
      U(i);
    }
    // fprintf(stderr, "\n");
  }
  void Lock1_Loop_0() { Lock1_Loop(0, 100000); }
  void Lock1_Loop_1() { Lock1_Loop(1, 100000); }
  void Lock1_Loop_2() { Lock1_Loop(2, 100000); }

  void CreateAndDestroyManyLocks() {
    LockType create_many_locks_but_never_acquire[kDeadlockGraphSize];
  }
  void CreateLockUnlockAndDestroyManyLocks() {
    LockType many_locks[kDeadlockGraphSize];
    for (size_t i = 0; i < kDeadlockGraphSize; i++) {
      many_locks[i].lock();
      many_locks[i].unlock();
    }
  }

  // LockTest Member function callback.
  struct CB {
    void (LockTest::*f)();
    LockTest *lt;
  };

  // Thread function with CB.
  static void *Thread(void *param) {
    CB *cb = (CB*)param;
    (cb->lt->*cb->f)();
    return NULL;
  }

  void RunThreads(void (LockTest::*f1)(), void (LockTest::*f2)(),
                  void (LockTest::*f3)() = 0) {
    const int kNumThreads = 3;
    pthread_t t[kNumThreads];
    CB cb[kNumThreads] = {{f1, this}, {f2, this}, {f3, this}};
    for (int i = 0; i < kNumThreads && cb[i].f; i++)
      pthread_create(&t[i], 0, Thread, &cb[i]);
    for (int i = 0; i < kNumThreads && cb[i].f; i++)
      pthread_join(t[i], 0);
  }

  static const size_t kDeadlockGraphSize = 4096;
  size_t n_;
  LockType *locks_;
};

template <class LockType>
void RunAllTests() {
  { LockTest<LockType> t(5); t.Test1(); }
  { LockTest<LockType> t(5); t.Test2(); }
  { LockTest<LockType> t(5); t.Test3(); }
  { LockTest<LockType> t(5); t.Test4(); }
  { LockTest<LockType> t(5); t.Test5(); }
  { LockTest<LockType> t(5); t.Test6(); }
  { LockTest<LockType> t(10); t.Test7(); }
}

int main () {
  RunAllTests<PthreadMutex>();
  RunAllTests<PthreadSpinLock>();
  fprintf(stderr, "DONE\n");
  // CHECK: DONE
}

