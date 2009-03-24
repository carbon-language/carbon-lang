// RUN: clang-cc -fsyntax-only %s

char *const_cast_test(const char *var)
{
  return const_cast<char*>(var);
}

#if 0
// FIXME: Uncomment when C++ is supported more.
struct A {
  virtual ~A() {}
};

struct B : public A {
};

struct B *dynamic_cast_test(struct A *a)
{
  return dynamic_cast<struct B*>(a);
}
#endif

char *reinterpret_cast_test()
{
  return reinterpret_cast<char*>(0xdeadbeef);
}

double static_cast_test(int i)
{
  return static_cast<double>(i);
}

char postfix_expr_test()
{
  return reinterpret_cast<char*>(0xdeadbeef)[0];
}
