
namespace foo {
  namespace bar {
    struct X { X(); };

    X::X() {}
  }
}


namespace {
  struct Y { Y(); };
  Y::Y() {}
}
