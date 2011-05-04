// RUN: %clang_cc1 -fsyntax-only -verify %s

// <rdar://problem/9168556>
typedef struct Point Point;

namespace NameSpace {
  class Point;
}

using namespace NameSpace;

class Test
{
public:
  struct Point { };
  virtual bool testMethod (Test::Point& p) = 0;
};
