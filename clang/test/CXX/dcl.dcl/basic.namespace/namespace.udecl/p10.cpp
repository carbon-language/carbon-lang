// RUN: clang -fsyntax-only -verify %s

namespace test0 {
  namespace ns0 {
    class tag;
    int tag();
  }

  namespace ns1 {
    using ns0::tag;
  }

  namespace ns2 {
    using ns0::tag;
  }

  using ns1::tag;
  using ns2::tag;
}
