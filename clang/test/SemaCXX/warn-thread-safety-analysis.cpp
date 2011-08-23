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
   // int x __attribute__((guarded_by(mu))); // FIXME: scoping error
};

MutexWrapper sls_mw;

void sls_fun_0() {
  sls_mw.mu.Lock();
  // sls_mw.x = 5; // FIXME: turn mu into sls_mw.mu
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

void sls_fun_bad_1() {
  sls_mu.Unlock(); // \
    expected-warning {{unlocking 'sls_mu' that was not acquired}}
}

void sls_fun_bad_2() {
  sls_mu.Lock();
  sls_mu.Lock(); // \
    expected-warning {{locking 'sls_mu' that is already acquired}}
  sls_mu.Unlock();
}

void sls_fun_bad_3() {
  sls_mu.Lock(); // \
    expected-warning {{lock 'sls_mu' is not released at the end of function 'sls_fun_bad_3'}}
}

void sls_fun_bad_4() {
  if (getBool())
    sls_mu.Lock(); // \
      expected-warning {{lock 'sls_mu' is not released at the end of its scope}}
  else
    sls_mu2.Lock(); // \
      expected-warning {{lock 'sls_mu2' is not released at the end of its scope}}
}

void sls_fun_bad_5() {
  sls_mu.Lock(); // \
    expected-warning {{lock 'sls_mu' is not released at the end of its scope}}
  if (getBool())
    sls_mu.Unlock();
}

void sls_fun_bad_6() {
  if (getBool()) {
    sls_mu.Lock(); // \
      expected-warning {{lock 'sls_mu' is not released at the end of its scope}}
  } else {
    if (getBool()) {
      getBool(); // EMPTY
    } else {
      getBool(); // EMPTY
    }
  }
  sls_mu.Unlock(); // \
    expected-warning {{unlocking 'sls_mu' that was not acquired}}
}

void sls_fun_bad_7() {
  sls_mu.Lock();
  while (getBool()) { // \
      expected-warning {{expecting lock 'sls_mu' to be held at start of each loop}}
    sls_mu.Unlock();
    if (getBool()) {
      if (getBool()) {
        continue;
      }
    }
    sls_mu.Lock(); // \
      expected-warning {{lock 'sls_mu' is not released at the end of its scope}}
  }
  sls_mu.Unlock();
}

void sls_fun_bad_8() {
  sls_mu.Lock();
  do {
    sls_mu.Unlock();  // \
      expected-warning {{expecting lock 'sls_mu' to be held at start of each loop}}
  } while (getBool());
}

void sls_fun_bad_9() {
  do {
    sls_mu.Lock(); // \
      expected-warning {{lock 'sls_mu' is not released at the end of its scope}}
  } while (getBool());
  sls_mu.Unlock();
}

void sls_fun_bad_10() {
  sls_mu.Lock(); // \
    expected-warning {{lock 'sls_mu' is not released at the end of function 'sls_fun_bad_10'}}
  while(getBool()) { // \
      expected-warning {{expecting lock 'sls_mu' to be held at start of each loop}}
    sls_mu.Unlock();
  }
}

void sls_fun_bad_11() {
  while (getBool()) {
    sls_mu.Lock(); // \
      expected-warning {{lock 'sls_mu' is not released at the end of its scope}}
  }
  sls_mu.Unlock(); // \
    expected-warning {{unlocking 'sls_mu' that was not acquired}}
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

void aa_elr_fun() __attribute__((exclusive_locks_required(aa_mu)));
void aa_elr_fun() { }

void aa_fun_1() {
  glock.globalLock();
  glock.globalUnlock();
}

void aa_fun_2() {
  aa_mu.Lock();
  aa_elr_fun();
  aa_mu.Unlock();
}

void aa_fun_bad_1() {
  glock.globalUnlock(); // \
    expected-warning {{unlocking 'aa_mu' that was not acquired}}
}

void aa_fun_bad_2() {
  glock.globalLock();
  glock.globalLock(); // \
    expected-warning {{locking 'aa_mu' that is already acquired}}
  glock.globalUnlock();
}

void aa_fun_bad_3() {
  glock.globalLock(); // \
    expected-warning {{lock 'aa_mu' is not released at the end of function 'aa_fun_bad_3'}}
}
