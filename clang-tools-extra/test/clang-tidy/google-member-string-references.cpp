// RUN: clang-tidy %s -checks='-*,google-runtime-member-string-references' -- | FileCheck %s -implicit-check-not="{{warning|error}}:"

namespace std {
template<typename T>
  class basic_string {};

typedef basic_string<char> string;
}

class string {};


struct A {
  const std::string &s;
// CHECK: :[[@LINE-1]]:3: warning: const string& members are dangerous. It is much better to use alternatives, such as pointers or simple constants.
};

struct B {
  std::string &s;
};

struct C {
  const std::string s;
};

template <typename T>
struct D {
  D();
  const T &s;
  const std::string &s2;
// CHECK: :[[@LINE-1]]:3: warning: const string& members are dangerous. It is much better to use alternatives, such as pointers or simple constants.
};

D<std::string> d;

struct AA {
  const string &s;
// CHECK: :[[@LINE-1]]:3: warning: const string& members are dangerous. It is much better to use alternatives, such as pointers or simple constants.
};

struct BB {
  string &s;
};

struct CC {
  const string s;
};

D<string> dd;
