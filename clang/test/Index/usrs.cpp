namespace foo {
  int x;
  void bar(int z);
}
namespace bar {
  typedef int QType;
  void bar(QType z);
}

class ClsA {
public:
  int a, b;
  ClsA(int A, int B) : a(A), b(B) {}
};

namespace foo {
  class ClsB : public ClsA {
  public:
    ClsB() : ClsA(1, 2) {}
    int result() const;
  };
}

int foo::ClsB::result() const {
  return a + b;
}

namespace {
  class ClsC : public foo::ClsB {};
  int w;
}

int z;

namespace foo { namespace taz {
  int x;
  static inline int add(int a, int b) { return a + b; }
  void sub(int a, int b);
}
}

namespace foo { namespace taz {
  class ClsD : public foo::ClsB {
  public:
    ClsD& operator=(int x) { a = x; return *this; }
    ClsD& operator=(double x) { a = (int) x; return *this; }
  };
}}

// RUN: c-index-test -test-load-source-usrs all %s | FileCheck %s
// CHECK: usrs.cpp c:@N@foo Extent=[1:11 - 4:2]
// CHECK: usrs.cpp c:@N@foo@x Extent=[2:3 - 2:8]
// CHECK: usrs.cpp c:@N@foo@F@bar Extent=[3:8 - 3:18]
// CHECK: usrs.cpp c:usrs.cpp@3:12@N@foo@F@bar@z Extent=[3:12 - 3:17]
// CHECK: usrs.cpp c:@N@bar Extent=[5:11 - 8:2]
// CHECK: usrs.cpp c:usrs.cpp@6:15@N@bar@T@QType Extent=[6:15 - 6:20]
// CHECK: usrs.cpp c:@N@bar@F@bar Extent=[7:8 - 7:20]
// CHECK: usrs.cpp c:usrs.cpp@7:12@N@bar@F@bar@z Extent=[7:12 - 7:19]
// CHECK: usrs.cpp c:@C@ClsA Extent=[10:1 - 14:2]
// CHECK: usrs.cpp c:@C@ClsA@FI@a Extent=[12:7 - 12:8]
// CHECK: usrs.cpp c:@C@ClsA@FI@b Extent=[12:10 - 12:11]
// CHECK: usrs.cpp c:@C@ClsA@F@ClsA Extent=[13:3 - 13:37]
// CHECK: usrs.cpp c:usrs.cpp@13:8@C@ClsA@F@ClsA@A Extent=[13:8 - 13:13]
// CHECK: usrs.cpp c:usrs.cpp@13:15@C@ClsA@F@ClsA@B Extent=[13:15 - 13:20]
// CHECK: usrs.cpp c:@N@foo Extent=[16:11 - 22:2]
// CHECK: usrs.cpp c:@N@foo@C@ClsB Extent=[17:3 - 21:4]
// CHECK: usrs.cpp c:@N@foo@C@ClsB@F@ClsB Extent=[19:5 - 19:27]
// CHECK: usrs.cpp c:@N@foo@C@ClsB@F@result Extent=[20:9 - 20:17]
// CHECK: usrs.cpp c:@N@foo@C@ClsB@F@result Extent=[24:16 - 26:2]
// CHECK: usrs.cpp c:@aN@C@ClsC Extent=[29:3 - 29:35]
// CHECK: usrs.cpp c:@aN@w Extent=[30:3 - 30:8]
// CHECK: usrs.cpp c:@z Extent=[33:1 - 33:6]
// CHECK: usrs.cpp c:@N@foo Extent=[35:11 - 40:2]
// CHECK: usrs.cpp c:@N@foo@N@taz Extent=[35:27 - 39:2]
// CHECK: usrs.cpp c:@N@foo@N@taz@x Extent=[36:3 - 36:8]
// CHECK: usrs.cpp c:usrs.cpp@37:21@N@foo@N@taz@F@add Extent=[37:21 - 37:56]
// CHECK: usrs.cpp c:usrs.cpp@37:25@N@foo@N@taz@F@add@a Extent=[37:25 - 37:30]
// CHECK: usrs.cpp c:usrs.cpp@37:32@N@foo@N@taz@F@add@b Extent=[37:32 - 37:37]
// CHECK: usrs.cpp c:@N@foo@N@taz@F@sub Extent=[38:8 - 38:25]
// CHECK: usrs.cpp c:usrs.cpp@38:12@N@foo@N@taz@F@sub@a Extent=[38:12 - 38:17]
// CHECK: usrs.cpp c:usrs.cpp@38:19@N@foo@N@taz@F@sub@b Extent=[38:19 - 38:24]
// CHECK: usrs.cpp c:@N@foo Extent=[42:11 - 48:3]
// CHECK: usrs.cpp c:@N@foo@N@taz Extent=[42:27 - 48:2]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD Extent=[43:3 - 47:4]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD@F@operator= Extent=[45:11 - 45:52]
// CHECK: usrs.cpp c:usrs.cpp@45:21@N@foo@N@taz@C@ClsD@F@operator=@x Extent=[45:21 - 45:26]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD@F@operator= Extent=[46:11 - 46:61]
// CHECK: usrs.cpp c:usrs.cpp@46:21@N@foo@N@taz@C@ClsD@F@operator=@x Extent=[46:21 - 46:29]



