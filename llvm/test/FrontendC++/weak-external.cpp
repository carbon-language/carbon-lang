// RUN: %llvmgxx %s -S -O2 -o - | not grep {_ZNSs12_S_constructIPKcEEPcT_S3_RKSaIcESt20forward_iterator_tag}
// PR4262

// The "basic_string" extern template instantiation declaration is supposed to
// suppress the implicit instantiation of non-inline member functions. Make sure
// that we suppress the implicit instantiation of non-inline member functions
// defined out-of-line. That we aren't instantiating the basic_string
// constructor when we shouldn't be. Such an instantiation forces the implicit
// instantiation of _S_construct<const char*>. Since _S_construct is a member
// template, it's instantiation is *not* suppressed (despite being in
// basic_string<char>), so we would emit it as a weak definition.

#include <stdexcept>

void dummysymbol() {
  throw(std::runtime_error("string"));
}
