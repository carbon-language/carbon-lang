// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify -DSUPPRESSED=1 %s

#ifdef SUPPRESSED
// expected-no-diagnostics
#endif

namespace rdar12676053 {
  // Delta-reduced from a preprocessed file.
  template<class T>
  class RefCount {
    T *ref;
  public:
    T *operator->() const {
      return ref ? ref : 0;
    }
  };

  class string {};

  class ParserInputState {
  public:
    string filename;
  };

  class Parser {
    void setFilename(const string& f)  {
      inputState->filename = f;
#ifndef SUPPRESSED
// expected-warning@-2 {{Called C++ object pointer is null}}
#endif
    }
  protected:
    RefCount<ParserInputState> inputState;
  };
}