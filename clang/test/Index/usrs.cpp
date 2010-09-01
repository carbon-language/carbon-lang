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
    ClsD& operator=(const ClsD &x) { a = x.a; return *this; }
    static int qux();
    static int uz(int z, ...);
    bool operator==(const ClsD &x) const { return a == x.a; }
  };
}}

extern "C" {
  void rez(int a, int b);
}

namespace foo_alias = foo;

using namespace foo;

namespace foo_alias2 = foo;

// RUN: c-index-test -test-load-source-usrs all %s | FileCheck %s
// CHECK: usrs.cpp c:@N@foo Extent=[1:11 - 4:2]
// CHECK: usrs.cpp c:@N@foo@x Extent=[2:3 - 2:8]
// CHECK: usrs.cpp c:@N@foo@F@bar#I# Extent=[3:8 - 3:18]
// CHECK: usrs.cpp c:usrs.cpp@36@N@foo@F@bar#I#@z Extent=[3:12 - 3:17]
// CHECK: usrs.cpp c:@N@bar Extent=[5:11 - 8:2]
// CHECK: usrs.cpp c:usrs.cpp@76@N@bar@T@QType Extent=[6:15 - 6:20]
// CHECK: usrs.cpp c:@N@bar@F@bar#I# Extent=[7:8 - 7:20]
// CHECK: usrs.cpp c:usrs.cpp@94@N@bar@F@bar#I#@z Extent=[7:12 - 7:19]
// CHECK: usrs.cpp c:@C@ClsA Extent=[10:1 - 14:2]
// CHECK: usrs.cpp c: Extent=[11:1 - 11:8]
// CHECK: usrs.cpp c:@C@ClsA@FI@a Extent=[12:7 - 12:8]
// CHECK: usrs.cpp c:@C@ClsA@FI@b Extent=[12:10 - 12:11]
// CHECK: usrs.cpp c:@C@ClsA@F@ClsA#I#I# Extent=[13:3 - 13:37]
// CHECK: usrs.cpp c:usrs.cpp@147@C@ClsA@F@ClsA#I#I#@A Extent=[13:8 - 13:13]
// CHECK: usrs.cpp c:usrs.cpp@154@C@ClsA@F@ClsA#I#I#@B Extent=[13:15 - 13:20]
// CHECK: usrs.cpp c:@N@foo Extent=[16:11 - 22:2]
// CHECK: usrs.cpp c:@N@foo@C@ClsB Extent=[17:3 - 21:4]
// CHECK: usrs.cpp c: Extent=[18:3 - 18:10]
// CHECK: usrs.cpp c:@N@foo@C@ClsB@F@ClsB# Extent=[19:5 - 19:27]
// CHECK: usrs.cpp c:@N@foo@C@ClsB@F@result#1 Extent=[20:9 - 20:17]
// CHECK: usrs.cpp c:@N@foo@C@ClsB@F@result#1 Extent=[24:16 - 26:2]
// CHECK: usrs.cpp c:@aN@C@ClsC Extent=[29:3 - 29:35]
// CHECK: usrs.cpp c:@aN@w Extent=[30:3 - 30:8]
// CHECK: usrs.cpp c:@z Extent=[33:1 - 33:6]
// CHECK: usrs.cpp c:@N@foo Extent=[35:11 - 40:2]
// CHECK: usrs.cpp c:@N@foo@N@taz Extent=[35:27 - 39:2]
// CHECK: usrs.cpp c:@N@foo@N@taz@x Extent=[36:3 - 36:8]
// CHECK: usrs.cpp c:usrs.cpp@475@N@foo@N@taz@F@add#I#I# Extent=[37:21 - 37:56]
// CHECK: usrs.cpp c:usrs.cpp@479@N@foo@N@taz@F@add#I#I#@a Extent=[37:25 - 37:30]
// CHECK: usrs.cpp c:usrs.cpp@486@N@foo@N@taz@F@add#I#I#@b Extent=[37:32 - 37:37]
// CHECK: usrs.cpp c:@N@foo@N@taz@F@sub#I#I# Extent=[38:8 - 38:25]
// CHECK: usrs.cpp c:usrs.cpp@522@N@foo@N@taz@F@sub#I#I#@a Extent=[38:12 - 38:17]
// CHECK: usrs.cpp c:usrs.cpp@529@N@foo@N@taz@F@sub#I#I#@b Extent=[38:19 - 38:24]
// CHECK: usrs.cpp c:@N@foo Extent=[42:11 - 52:3]
// CHECK: usrs.cpp c:@N@foo@N@taz Extent=[42:27 - 52:2]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD Extent=[43:3 - 51:4]
// CHECK: usrs.cpp c: Extent=[44:3 - 44:10]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD@F@operator=#I# Extent=[45:11 - 45:52]
// CHECK: usrs.cpp c:usrs.cpp@638@N@foo@N@taz@C@ClsD@F@operator=#I#@x Extent=[45:21 - 45:26]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD@F@operator=#d# Extent=[46:11 - 46:61]
// CHECK: usrs.cpp c:usrs.cpp@690@N@foo@N@taz@C@ClsD@F@operator=#d#@x Extent=[46:21 - 46:29]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD@F@operator=#&1$@N@foo@N@taz@C@ClsD# Extent=[47:11 - 47:62]
// CHECK: usrs.cpp c:usrs.cpp@757@N@foo@N@taz@C@ClsD@F@operator=#&1$@N@foo@N@taz@C@ClsD#@x Extent=[47:27 - 47:34]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD@F@qux#S Extent=[48:16 - 48:21]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD@F@uz#I.#S Extent=[49:16 - 49:30]
// CHECK: usrs.cpp c:usrs.cpp@833@N@foo@N@taz@C@ClsD@F@uz#I.#S@z Extent=[49:19 - 49:24]
// CHECK: usrs.cpp c:@N@foo@N@taz@C@ClsD@F@operator==#&1$@N@foo@N@taz@C@ClsD#1 Extent=[50:10 - 50:62]
// CHECK: usrs.cpp c:usrs.cpp@872@N@foo@N@taz@C@ClsD@F@operator==#&1$@N@foo@N@taz@C@ClsD#1@x Extent=[50:27 - 50:34]
// CHECK: usrs.cpp c:@F@rez Extent=[55:8 - 55:25]
// CHECK: usrs.cpp c:usrs.cpp@941@F@rez@a Extent=[55:12 - 55:17]
// CHECK: usrs.cpp c:usrs.cpp@948@F@rez@b Extent=[55:19 - 55:24]
// CHECK: usrs.cpp c:@NA@foo_alias
// CHECK-NOT: foo
// CHECK: usrs.cpp c:@NA@foo_alias2
