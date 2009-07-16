// RUN: %llvmgxx %s -S
// XFAIL: darwin

#include <set>

class A {
public:
  A();
private:
  A(const A&);
};
void B()
{
  std::set<void *, A> foo;
}
