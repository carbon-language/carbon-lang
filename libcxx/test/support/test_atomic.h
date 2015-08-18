#ifndef SUPPORT_TEST_ATOMIC_H
#define SUPPORT_TEST_ATOMIC_H

// If the atomic memory order macros are defined then assume
// the compiler supports the required atomic builtins.
#if !defined(__ATOMIC_SEQ_CST)
#define TEST_HAS_NO_ATOMICS
#endif

template <class ValType>
class Atomic {
   ValType value;
   Atomic(Atomic const&);
   Atomic& operator=(Atomic const&);
   Atomic& operator=(Atomic const&) volatile;
private:
  enum {
#if !defined(TEST_HAS_NO_ATOMICS)
    AO_Relaxed = __ATOMIC_RELAXED,
    AO_Seq     = __ATOMIC_SEQ_CST
#else
    AO_Relaxed,
    AO_Seq
#endif
  };
  template <class Tp, class FromType>
  static inline void atomic_store_imp(Tp* dest, FromType from, int order = AO_Seq) {
#if !defined(TEST_HAS_NO_ATOMICS)
      __atomic_store_n(dest, from, order);
#else
    *dest = from;
#endif
  }

  template <class Tp>
  static inline Tp atomic_load_imp(Tp* from, int order = AO_Seq) {
#if !defined(TEST_HAS_NO_ATOMICS)
    return __atomic_load_n(from, order);
#else
    return *from;
#endif
  }

  template <class Tp, class AddType>
  static inline Tp atomic_add_imp(Tp* val, AddType add, int order = AO_Seq) {
#if !defined(TEST_HAS_NO_ATOMICS)
      return __atomic_add_fetch(val, add, order);
#else
    return *val += add;
#endif
  }

  template <class Tp>
  static inline Tp atomic_exchange_imp(Tp* val, Tp other, int order = AO_Seq) {
#if !defined(TEST_HAS_NO_ATOMICS)
      return __atomic_exchange_n(val, other, order);
#else
      Tp old = *val;
      *val = other;
      return old;
#endif
  }
public:
    Atomic() : value(0) {}
    Atomic(ValType x) : value(x) {}

    ValType operator=(ValType val) {
       atomic_store_imp(&value, val);
       return val;
    }

    ValType operator=(ValType val) volatile {
        atomic_store_imp(&value, val);
        return val;
    }

    ValType load() const volatile { return atomic_load_imp(&value); }
    void    store(ValType val) volatile { atomic_store_imp(&value, val); }

    ValType relaxedLoad() const volatile { return atomic_load_imp(&value, AO_Relaxed); }
    void    relaxedStore(ValType val) volatile { atomic_store_imp(&value, val, AO_Relaxed); }

    ValType exchange(ValType other) volatile { return atomic_exchange_imp(&value, other); }
    bool    testAndSet() volatile { return atomic_exchange_imp(&value, 1); }
    void    clear() volatile { atomic_store_imp(&value, 0); }

    operator ValType() const { return atomic_load_imp(&value); }
    operator ValType() const volatile { return atomic_load_imp(&value); }

    ValType operator+=(ValType val)  { return atomic_add_imp(&value, val); }
    ValType operator-=(ValType val)  { return atomic_add_imp(&value, -val); }
    ValType operator+=(ValType val) volatile { return atomic_add_imp(&value, val); }
    ValType operator-=(ValType val) volatile { return atomic_add_imp(&value, -val); }

    ValType operator++()  { return *this += 1; }
    ValType operator++(int) { return (*this += 1) - 1;  }
    ValType operator++() volatile { return *this += 1; }
    ValType operator++(int) volatile { return (*this += 1) - 1;  }

    ValType operator--() { return *this -= 1; }
    ValType operator--(int) { return (*this -= 1) + 1; }
    ValType operator--() volatile { return *this -= 1; }
    ValType operator--(int) volatile { return (*this -= 1) + 1; }
};

typedef Atomic<int> AtomicInt;
typedef Atomic<bool> AtomicBool;

#endif // SUPPORT_TEST_ATOMIC_H
