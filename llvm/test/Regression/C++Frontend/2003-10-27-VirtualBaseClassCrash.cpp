
template<class T>
struct super {
  int Y;
  void foo();
};

template <class T> 
struct test : virtual super<int> {};

extern test<int> X;

void foo() {
  X.foo();
}
