// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR4103 : Make sure we don't get a bogus unused expression warning
class APInt {
  char foo;
};
class APSInt : public APInt {
  char bar;
public:
  APSInt &operator=(const APSInt &RHS);
};

APSInt& APSInt::operator=(const APSInt &RHS) {
  APInt::operator=(RHS);
  return *this;
}
