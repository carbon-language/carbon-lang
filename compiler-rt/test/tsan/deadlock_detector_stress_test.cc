// RUN: %clangxx_tsan %s -o %t
// RUN: TSAN_OPTIONS=detect_deadlocks=1 %t 2>&1 | FileCheck %s
#include <pthread.h>
#undef NDEBUG
#include <assert.h>
#include <stdio.h>

class PaddedLock {
 public:
  PaddedLock() { assert(0 == pthread_mutex_init(&mu_, 0)); }
  ~PaddedLock() {
    assert(0 == pthread_mutex_destroy(&mu_));
    (void)padding_;
  }
  void lock() { assert(0 == pthread_mutex_lock(&mu_)); }
  void unlock() { assert(0 == pthread_mutex_unlock(&mu_)); }

 private:
  pthread_mutex_t mu_;
  char padding_[64 - sizeof(pthread_mutex_t)];
};

class LockTest {
 public:
  LockTest(size_t n) : n_(n), locks_(new PaddedLock[n]) { }
  ~LockTest() { delete [] locks_; }
  void L(size_t i) {
    assert(i < n_);
    locks_[i].lock();
  }
  void U(size_t i) {
    assert(i < n_);
    locks_[i].unlock();
  }

  void Test1() {
    fprintf(stderr, "Starting Test1\n");
    // CHECK: Starting Test1
    L(0); L(1); U(0); U(1);
    L(1); L(0); U(0); U(1);
    // CHECK: ThreadSanitizer: lock-order-inversion (potential deadlock)
    // CHECK-NOT: ThreadSanitizer:
  }

  void Test2() {
    fprintf(stderr, "Starting Test2\n");
    // CHECK: Starting Test2
    L(0); L(1); L(2); U(2); U(0); U(1);
    L(2); L(0); U(0); U(2);
    // CHECK: ThreadSanitizer: lock-order-inversion (potential deadlock)
    // CHECK-NOT: ThreadSanitizer:
  }

 private:
  size_t n_;
  PaddedLock *locks_;
};

int main() {
  { LockTest t(5); t.Test1(); }
  { LockTest t(5); t.Test2(); }
  fprintf(stderr, "DONE\n");
  // CHECK: DONE
}

