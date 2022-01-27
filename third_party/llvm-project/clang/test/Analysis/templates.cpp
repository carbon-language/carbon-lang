// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -fblocks -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -fblocks -analyzer-config c++-template-inlining=false -DNO_INLINE -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);

// Do not crash on this templated code which uses a block.
typedef void (^my_block)(void);
static void useBlock(my_block block){}
template<class T> class MyClass;
typedef MyClass<float> Mf;

template<class T>
class MyClass
{
public:
  MyClass() {}
  MyClass(T a);
  void I();
private:
 static const T one;
};

template<class T> const T MyClass<T>::one = static_cast<T>(1);
template<class T> inline MyClass<T>::MyClass(T a){}
template<class T> void MyClass<T>::I() {
  static MyClass<T>* mPtr = 0;
  useBlock(^{ mPtr = new MyClass<T> (MyClass<T>::one); });
};
int main(){
  Mf m;
  m.I();
}


// <rdar://problem/11949235>
template<class T, unsigned N>
inline unsigned array_lengthof(T (&)[N]) {
  return N;
}

void testNonTypeTemplateInstantiation() {
  const char *S[] = { "a", "b" };
  clang_analyzer_eval(array_lengthof(S) == 2);
#ifndef NO_INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif
}

namespace rdar13954714 {
  template <bool VALUE>
  bool blockInTemplate() {
    return (^() {
      return VALUE;
    })();
  }

  // force instantiation
  template bool blockInTemplate<true>();

  template <bool VALUE>
  void blockWithStatic() {
    (void)^() {
      static int x;
      return ++x;
    };
  }

  // force instantiation
  template void blockWithStatic<true>();
}
