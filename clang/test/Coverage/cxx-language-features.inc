//-*- C++ -*-

// Intended to exercise all syntactic parts of the C++ language that
// aren't part of C.

namespace std {
  namespace debug {
  }
}

using namespace std::debug;
using namespace std;

namespace safestl = ::std::debug;

class Base1 { 
};

class Base2 { };

class Derived1 : Base1, virtual public Base2 { };
