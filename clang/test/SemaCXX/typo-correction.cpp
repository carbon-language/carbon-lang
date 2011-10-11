// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c++0x-extensions %s

struct errc {
  int v_;
  operator int() const {return v_;}
};

class error_condition
{
  int _val_;
public:
  error_condition() : _val_(0) {}

  error_condition(int _val)
    : _val_(_val) {}

  template <class E>
  error_condition(E _e) {
    // make_error_condition must not be typo corrected to error_condition
    // even though the first declaration of make_error_condition has not
    // yet been encountered. This was a bug in the first version of the type
    // name typo correction patch that wasn't noticed until building LLVM with
    // Clang failed.
    *this = make_error_condition(_e);
  }

};

inline error_condition make_error_condition(errc _e) {
  return error_condition(static_cast<int>(_e));
}
