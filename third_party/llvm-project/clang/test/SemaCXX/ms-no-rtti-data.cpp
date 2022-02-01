// RUN: %clang_cc1 %s -triple x86_64-windows-msvc -fdiagnostics-format msvc -fno-rtti-data -fsyntax-only -verify

namespace std {
struct type_info {};
} // namespace std
class B {
public:
  virtual ~B() = default;
};

class D1 : public B {
public:
  ~D1() = default;
};

void f() {
  B *b = new D1();
  auto d = dynamic_cast<D1 *>(b);    // expected-warning{{dynamic_cast will not work since RTTI data is disabled by /GR-}}
  void *v = dynamic_cast<void *>(b); // expected-warning{{dynamic_cast will not work since RTTI data is disabled by /GR-}}

  (void)typeid(int);
  (void)typeid(b);
  (void)typeid(*b); // expected-warning{{typeid will not work since RTTI data is disabled by /GR-}}
  B b2 = *b;
  (void)typeid(b2);
  (void)typeid(*&b2); // expected-warning{{typeid will not work since RTTI data is disabled by /GR-}}
  (void)typeid((B &)b2);

  B &br = b2;
  (void)typeid(br); // expected-warning{{typeid will not work since RTTI data is disabled by /GR-}}
  (void)typeid(&br);
}