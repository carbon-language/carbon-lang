// RUN: %clang_cc1 -fsyntax-only -verify %s

@class NSString;

// Reduced from WebKit.
namespace test0 {
  template <typename T> struct RemovePointer {
    typedef T Type;
  };

  template <typename T> struct RemovePointer<T*> {
    typedef T Type;
  };

  template <typename T> struct RetainPtr {
    typedef typename RemovePointer<T>::Type ValueType;
    typedef ValueType* PtrType;
    RetainPtr(PtrType ptr);
  };
 
  void test(NSString *S) {
    RetainPtr<NSString*> ptr(S);
  }
}
