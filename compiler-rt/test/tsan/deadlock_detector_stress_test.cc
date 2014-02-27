// RUN: %clangxx_tsan %s -o %t -DLockType=PthreadMutex
// RUN: TSAN_OPTIONS=detect_deadlocks=1 not %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan %s -o %t -DLockType=PthreadSpinLock
// RUN: TSAN_OPTIONS=detect_deadlocks=1 not %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan %s -o %t -DLockType=PthreadRWLock
// RUN: TSAN_OPTIONS=detect_deadlocks=1 not %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-RD
// RUN: %clangxx_tsan %s -o %t -DLockType=PthreadRecursiveMutex
// RUN: TSAN_OPTIONS=detect_deadlocks=1 not %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-REC
#include <pthread.h>
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef LockType
#define LockType PthreadMutex
#endif

// You can optionally pass [test_number [iter_count]] on command line.
static int test_number = -1;
static int iter_count = 100000;

class PthreadMutex {
 public:
  explicit PthreadMutex(bool recursive = false) {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    if (recursive)
      pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    assert(0 == pthread_mutex_init(&mu_, &attr));
  }
  ~PthreadMutex() {
    assert(0 == pthread_mutex_destroy(&mu_));
    (void)padding_;
  }
  static bool supports_read_lock() { return false; }
  static bool supports_recursive_lock() { return false; }
  void lock() { assert(0 == pthread_mutex_lock(&mu_)); }
  void unlock() { assert(0 == pthread_mutex_unlock(&mu_)); }
  bool try_lock() { return 0 == pthread_mutex_trylock(&mu_); }
  void rdlock() { assert(0); }
  void rdunlock() { assert(0); }
  bool try_rdlock() { assert(0); }

 private:
  pthread_mutex_t mu_;
  char padding_[64 - sizeof(pthread_mutex_t)];
};

class PthreadRecursiveMutex : public PthreadMutex {
 public:
  PthreadRecursiveMutex() : PthreadMutex(true) { }
  static bool supports_recursive_lock() { return true; }
};

class PthreadSpinLock {
 public:
  PthreadSpinLock() { assert(0 == pthread_spin_init(&mu_, 0)); }
  ~PthreadSpinLock() {
    assert(0 == pthread_spin_destroy(&mu_));
    (void)padding_;
  }
  static bool supports_read_lock() { return false; }
  static bool supports_recursive_lock() { return false; }
  void lock() { assert(0 == pthread_spin_lock(&mu_)); }
  void unlock() { assert(0 == pthread_spin_unlock(&mu_)); }
  bool try_lock() { return 0 == pthread_spin_trylock(&mu_); }
  void rdlock() { assert(0); }
  void rdunlock() { assert(0); }
  bool try_rdlock() { assert(0); }

 private:
  pthread_spinlock_t mu_;
  char padding_[64 - sizeof(pthread_spinlock_t)];
};

class PthreadRWLock {
 public:
  PthreadRWLock() { assert(0 == pthread_rwlock_init(&mu_, 0)); }
  ~PthreadRWLock() {
    assert(0 == pthread_rwlock_destroy(&mu_));
    (void)padding_;
  }
  static bool supports_read_lock() { return true; }
  static bool supports_recursive_lock() { return false; }
  void lock() { assert(0 == pthread_rwlock_wrlock(&mu_)); }
  void unlock() { assert(0 == pthread_rwlock_unlock(&mu_)); }
  bool try_lock() { return 0 == pthread_rwlock_trywrlock(&mu_); }
  void rdlock() { assert(0 == pthread_rwlock_rdlock(&mu_)); }
  void rdunlock() { assert(0 == pthread_rwlock_unlock(&mu_)); }
  bool try_rdlock() { return 0 == pthread_rwlock_tryrdlock(&mu_); }

 private:
  pthread_rwlock_t mu_;
  char padding_[64 - sizeof(pthread_rwlock_t)];
};

class LockTest {
 public:
  LockTest() : n_(), locks_() {}
  void Init(size_t n) {
    n_ = n;
    locks_ = new LockType*[n_];
    for (size_t i = 0; i < n_; i++)
      locks_[i] = new LockType;
  }
  ~LockTest() {
    for (size_t i = 0; i < n_; i++)
      delete locks_[i];
    delete [] locks_;
  }
  void L(size_t i) {
    assert(i < n_);
    locks_[i]->lock();
  }

  void U(size_t i) {
    assert(i < n_);
    locks_[i]->unlock();
  }

  void RL(size_t i) {
    assert(i < n_);
    locks_[i]->rdlock();
  }

  void RU(size_t i) {
    assert(i < n_);
    locks_[i]->rdunlock();
  }

  void *A(size_t i) {
    assert(i < n_);
    return locks_[i];
  }

  bool T(size_t i) {
    assert(i < n_);
    return locks_[i]->try_lock();
  }

  // Simple lock order onversion.
  void Test1() {
    if (test_number > 0 && test_number != 1) return;
    fprintf(stderr, "Starting Test1\n");
    // CHECK: Starting Test1
    Init(5);
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
    if (test_number > 0 && test_number != 2) return;
    fprintf(stderr, "Starting Test2\n");
    // CHECK: Starting Test2
    Init(5);
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
    if (test_number > 0 && test_number != 3) return;
    fprintf(stderr, "Starting Test3\n");
    // CHECK: Starting Test3
    Init(5);
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
    if (test_number > 0 && test_number != 4) return;
    fprintf(stderr, "Starting Test4\n");
    // CHECK: Starting Test4
    Init(5);
    Lock_0_1();
    L(2);
    CreateLockUnlockAndDestroyManyLocks();
    U(2);
    Lock_1_0();
    // CHECK-NOT: WARNING: ThreadSanitizer:
  }

  void Test5() {
    if (test_number > 0 && test_number != 5) return;
    fprintf(stderr, "Starting Test5\n");
    // CHECK: Starting Test5
    Init(5);
    RunThreads(&LockTest::Lock_0_1, &LockTest::Lock_1_0);
    // CHECK: WARNING: ThreadSanitizer: lock-order-inversion
    // CHECK-NOT: WARNING: ThreadSanitizer:
  }

  void Test6() {
    if (test_number > 0 && test_number != 6) return;
    fprintf(stderr, "Starting Test6: 3 threads lock/unlock private mutexes\n");
    // CHECK: Starting Test6
    Init(100);
    // CHECK-NOT: WARNING: ThreadSanitizer:
    RunThreads(&LockTest::Lock1_Loop_0, &LockTest::Lock1_Loop_1,
               &LockTest::Lock1_Loop_2);
  }

  void Test7() {
    if (test_number > 0 && test_number != 7) return;
    fprintf(stderr, "Starting Test7\n");
    // CHECK: Starting Test7
    Init(10);
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

  void Test8() {
    if (test_number > 0 && test_number != 8) return;
    if (!LockType::supports_read_lock()) return;
    fprintf(stderr, "Starting Test8\n");
    Init(5);
    // CHECK-RD: Starting Test8
    RL(0); L(1); RU(0); U(1);
    L(1); RL(0); RU(0); U(1);
    // CHECK-RD: WARNING: ThreadSanitizer: lock-order-inversion
    fprintf(stderr, "Have cycle: 0=>1\n");
    // CHECK-RD: Have cycle: 0=>1

    RL(2); RL(3); RU(2); RU(3);
    RL(3); RL(2); RU(2); RU(3);
    // CHECK-RD: WARNING: ThreadSanitizer: lock-order-inversion
    fprintf(stderr, "Have cycle: 2=>3\n");
    // CHECK-RD: Have cycle: 2=>3
  }

  void Test9() {
    if (test_number > 0 && test_number != 9) return;
    if (!LockType::supports_recursive_lock()) return;
    fprintf(stderr, "Starting Test9\n");
    // CHECK-REC: Starting Test9
    Init(5);
    L(0); L(0); L(0); L(1); U(1); U(0); U(0); U(0);
    L(1); L(1); L(1); L(0); U(0); U(1); U(1); U(1);
    // CHECK-REC: WARNING: ThreadSanitizer: lock-order-inversion
  }

  void Test10() {
    if (test_number > 0 && test_number != 10) return;
    fprintf(stderr, "Starting Test10: 4 threads lock/unlock 4 private mutexes, one under another\n");
    // CHECK: Starting Test10
    Init(100);
    // CHECK-NOT: WARNING: ThreadSanitizer:
    RunThreads(&LockTest::Test10_Thread1, &LockTest::Test10_Thread2,
               &LockTest::Test10_Thread3, &LockTest::Test10_Thread4);
  }
  void Test10_Thread1() { Test10_Thread(0); }
  void Test10_Thread2() { Test10_Thread(10); }
  void Test10_Thread3() { Test10_Thread(20); }
  void Test10_Thread4() { Test10_Thread(30); }
  void Test10_Thread(size_t m) {
    for (int i = 0; i < iter_count; i++) {
      L(m + 0);
      L(m + 1);
      L(m + 2);
      L(m + 3);
      U(m + 3);
      U(m + 2);
      U(m + 1);
      U(m + 0);
    }
  }

  void Test11() {
    if (test_number > 0 && test_number != 11) return;
    fprintf(stderr, "Starting Test11: 4 threads lock/unlock 4 private mutexes, all under another private mutex\n");
    // CHECK: Starting Test11
    Init(500);
    // CHECK-NOT: WARNING: ThreadSanitizer:
    RunThreads(&LockTest::Test11_Thread1, &LockTest::Test11_Thread2,
               &LockTest::Test11_Thread3, &LockTest::Test11_Thread4);
  }
  void Test11_Thread1() { Test10_Thread(0); }
  void Test11_Thread2() { Test10_Thread(10); }
  void Test11_Thread3() { Test10_Thread(20); }
  void Test11_Thread4() { Test10_Thread(30); }
  void Test11_Thread(size_t m) {
    for (int i = 0; i < iter_count; i++) {
      L(m);
      L(m + 100);
      U(m + 100);
      L(m + 200);
      U(m + 200);
      L(m + 300);
      U(m + 300);
      L(m + 400);
      U(m + 500);
      U(m);
    }
  }

  void Test12() {
    if (test_number > 0 && test_number != 12) return;
    if (!LockType::supports_read_lock()) return;
    fprintf(stderr, "Starting Test12: 4 threads read lock/unlock 4 shared mutexes, one under another\n");
    // CHECK-RD: Starting Test12
    Init(500);
    // CHECK-RD-NOT: WARNING: ThreadSanitizer:
    RunThreads(&LockTest::Test12_Thread, &LockTest::Test12_Thread,
               &LockTest::Test12_Thread, &LockTest::Test12_Thread);
  }
  void Test12_Thread() {
    for (int i = 0; i < iter_count; i++) {
      RL(000);
      RL(100);
      RL(200);
      RL(300);
      RU(300);
      RU(200);
      RU(100);
      RU(000);
    }
  }

  void Test13() {
    if (test_number > 0 && test_number != 13) return;
    if (!LockType::supports_read_lock()) return;
    fprintf(stderr, "Starting Test13: 4 threads read lock/unlock 4 shared mutexes, all under another shared mutex\n");
    // CHECK-RD: Starting Test13
    Init(500);
    // CHECK-RD-NOT: WARNING: ThreadSanitizer:
    RunThreads(&LockTest::Test13_Thread, &LockTest::Test13_Thread,
               &LockTest::Test13_Thread, &LockTest::Test13_Thread);
  }
  void Test13_Thread() {
    for (int i = 0; i < iter_count; i++) {
      RL(0);
      RL(100);
      RU(100);
      RL(200);
      RU(200);
      RL(300);
      RU(300);
      RL(400);
      RU(400);
      RU(0);
    }
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
  void Lock1_Loop_0() { Lock1_Loop(0, iter_count); }
  void Lock1_Loop_1() { Lock1_Loop(10, iter_count); }
  void Lock1_Loop_2() { Lock1_Loop(20, iter_count); }

  void CreateAndDestroyManyLocks() {
    LockType create_many_locks_but_never_acquire[kDeadlockGraphSize];
    (void)create_many_locks_but_never_acquire;
    (void)create_many_locks_but_never_acquire;
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
                  void (LockTest::*f3)() = 0, void (LockTest::*f4)() = 0) {
    const int kNumThreads = 4;
    pthread_t t[kNumThreads];
    CB cb[kNumThreads] = {{f1, this}, {f2, this}, {f3, this}, {f4, this}};
    for (int i = 0; i < kNumThreads && cb[i].f; i++)
      pthread_create(&t[i], 0, Thread, &cb[i]);
    for (int i = 0; i < kNumThreads && cb[i].f; i++)
      pthread_join(t[i], 0);
  }

  static const size_t kDeadlockGraphSize = 4096;
  size_t n_;
  LockType **locks_;
};

int main(int argc, char **argv) {
  if (argc > 1)
    test_number = atoi(argv[1]);
  if (argc > 2)
    iter_count = atoi(argv[2]);
  LockTest().Test1();
  LockTest().Test2();
  LockTest().Test3();
  LockTest().Test4();
  LockTest().Test5();
  LockTest().Test6();
  LockTest().Test7();
  LockTest().Test8();
  LockTest().Test9();
  LockTest().Test10();
  LockTest().Test11();
  LockTest().Test12();
  LockTest().Test13();
  fprintf(stderr, "ALL-DONE\n");
  // CHECK: ALL-DONE
}

