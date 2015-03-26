struct A {};
class B {
  struct Inner1 {};
  struct Inner2;
};
struct B::Inner2 : Inner1 {};
