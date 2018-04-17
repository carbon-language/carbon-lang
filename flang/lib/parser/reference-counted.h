#ifndef FORTRAN_PARSER_REFERENCE_COUNTED_H_
#define FORTRAN_PARSER_REFERENCE_COUNTED_H_

// A template class of smart pointers to objects with their own
// reference counting object lifetimes that's lighter weight
// than std::shared_ptr<>.  Not thread-safe.

namespace Fortran {
namespace parser {

template<typename A> class ReferenceCounted {
public:
  ReferenceCounted() {}
  void TakeReference() { ++references_; }
  void DropReference() {
    if (--references_ == 0) {
      delete static_cast<A*>(this);
    }
  }
private:
  int references_{0};
};

template<typename A> class CountedReference {
public:
  using type = A;
  CountedReference() {}
  CountedReference(type *m) : p_{m} { Take(); }
  CountedReference(const CountedReference &c) : p_{c.p_} { Take(); }
  CountedReference(CountedReference &&c) : p_{c.p_} { c.p_ = nullptr; }
  CountedReference &operator=(const CountedReference &c) {
    c.Take();
    Drop();
    p_ = c.p_;
    return *this;
  }
  CountedReference &operator=(CountedReference &&c) {
    A *p{c.p_};
    c.p_ = nullptr;
    Drop();
    p_ = p;
    return *this;
  }
  ~CountedReference() { Drop(); }
  operator bool() const { return p_ != nullptr; }
  type *get() const { return p_; }
  type &operator*() const { return *p_; }
  type *operator->() const { return p_; }

private:
  void Take() const {
    if (p_) {
      p_->TakeReference();
    }
  }
  void Drop() {
    if (p_) {
      p_->DropReference();
      p_ = nullptr;
    }
  }

  type *p_{nullptr};
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_REFERENCE_COUNTED_H_
