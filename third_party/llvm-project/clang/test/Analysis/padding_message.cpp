// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux -std=c++14 -analyzer-checker=optin.performance -analyzer-config optin.performance.Padding:AllowedPad=2 -verify %s

// expected-warning@+7{{\
Excessive padding in 'struct IntSandwich' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
i, \
c1, \
c2, \
}}
struct IntSandwich {
  char c1;
  int i;
  char c2;
};

// expected-warning@+7{{\
Excessive padding in 'struct TurDuckHen' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
i, \
c1, \
c2, \
}}
struct TurDuckHen {
  char c1;
  struct IntSandwich i;
  char c2;
};

#pragma pack(push)
#pragma pack(2)
// expected-warning@+11{{\
Excessive padding in 'struct SmallIntSandwich' (4 padding bytes, where 0 is optimal). \
Optimal fields order: \
i1, \
i2, \
i3, \
c1, \
c2, \
c3, \
c4, \
}}
struct SmallIntSandwich {
  char c1;
  int i1;
  char c2;
  int i2;
  char c3;
  int i3;
  char c4;
};
#pragma pack(pop)

union SomeUnion { // no-warning
  char c;
  short s;
  int i;
};

// expected-warning@+7{{\
Excessive padding in 'struct HoldsAUnion' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
u, \
c1, \
c2, \
}}
struct HoldsAUnion {
  char c1;
  union SomeUnion u;
  char c2;
};

struct SmallCharArray { // no-warning
  char c[5];
};

struct MediumIntArray { // no-warning
  int i[5];
};

// expected-warning@+7{{\
Excessive padding in 'struct StructSandwich' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
m, \
s, \
s2, \
}}
struct StructSandwich {
  struct SmallCharArray s;
  struct MediumIntArray m;
  struct SmallCharArray s2;
};

// expected-warning@+7{{\
Excessive padding in 'TypedefSandwich' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
i, \
c1, \
c2, \
}}
typedef struct {
  char c1;
  int i;
  char c2;
} TypedefSandwich;

// expected-warning@+7{{\
Excessive padding in 'struct StructAttrAlign' (10 padding bytes, where 2 is optimal). \
Optimal fields order: \
i, \
c1, \
c2, \
}}
struct StructAttrAlign {
  char c1;
  int i;
  char c2;
} __attribute__((aligned(8)));

// expected-warning@+8{{\
Excessive padding in 'struct OverlyAlignedChar' (8185 padding bytes, where 4089 is optimal). \
Optimal fields order: \
c, \
c1, \
c2, \
x, \
}}
struct OverlyAlignedChar {
  char c1;
  int x;
  char c2;
  char c __attribute__((aligned(4096)));
};

// expected-warning@+7{{\
Excessive padding in 'struct HoldsOverlyAlignedChar' (8190 padding bytes, where 4094 is optimal). \
Optimal fields order: \
o, \
c1, \
c2, \
}}
struct HoldsOverlyAlignedChar {
  char c1;
  struct OverlyAlignedChar o;
  char c2;
};

void internalStructFunc() {
  // expected-warning@+7{{\
Excessive padding in 'struct X' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
t, \
c1, \
c2, \
}}
  struct X {
    char c1;
    int t;
    char c2;
  };
  struct X obj;
}

void typedefStructFunc() {
  // expected-warning@+7{{\
Excessive padding in 'S' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
t, \
c1, \
c2, \
}}
  typedef struct {
    char c1;
    int t;
    char c2;
  } S;
  S obj;
}

// expected-warning@+7{{\
Excessive padding in 'struct DefaultAttrAlign' (22 padding bytes, where 6 is optimal). \
Optimal fields order: \
i, \
c1, \
c2, \
}}
struct DefaultAttrAlign {
  char c1;
  long long i;
  char c2;
} __attribute__((aligned));

// expected-warning@+7{{\
Excessive padding in 'struct SmallArrayShortSandwich' (2 padding bytes, where 0 is optimal). \
Optimal fields order: \
s, \
c1, \
c2, \
}}
struct SmallArrayShortSandwich {
  char c1;
  short s;
  char c2;
} ShortArray[20];

// expected-warning@+7{{\
Excessive padding in 'struct SmallArrayInFunc' (2 padding bytes, where 0 is optimal). \
Optimal fields order: \
s, \
c1, \
c2, \
}}
struct SmallArrayInFunc {
  char c1;
  short s;
  char c2;
};

void arrayHolder() {
  struct SmallArrayInFunc Arr[15];
}

// expected-warning@+7{{\
Excessive padding in 'class VirtualIntSandwich' (10 padding bytes, where 2 is optimal). \
Optimal fields order: \
i, \
c1, \
c2, \
}}
class VirtualIntSandwich {
  virtual void foo() {}
  char c1;
  int i;
  char c2;
};

// constructed so as not to have tail padding
// expected-warning@+8{{\
Excessive padding in 'class InnerPaddedB' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
i1, \
i2, \
c1, \
c2, \
}}
class InnerPaddedB {
  char c1;
  int i1;
  char c2;
  int i2;
};

class Empty {}; // no-warning

// expected-warning@+7{{\
Excessive padding in 'class LotsOfSpace' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
i, \
e1, \
e2, \
}}
class LotsOfSpace {
  Empty e1;
  int i;
  Empty e2;
};

// expected-warning@+7{{\
Excessive padding in 'struct TypedefSandwich2' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
t, \
c1, \
c2, \
}}
typedef struct TypedefSandwich2 {
  char c1;
  // expected-warning@+7{{\
Excessive padding in 'TypedefSandwich2::NestedTypedef' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
i, \
c1, \
c2, \
}}
  typedef struct {
    char c1;
    int i;
    char c2;
  } NestedTypedef;
  NestedTypedef t;
  char c2;
} TypedefSandwich2;

template <typename T>
struct Foo {
  // expected-warning@+7{{\
Excessive padding in 'struct Foo<int>::Nested' (6 padding bytes, where 2 is optimal). \
Optimal fields order: \
t, \
c1, \
c2, \
}}
  struct Nested {
    char c1;
    T t;
    char c2;
  };
};

struct Holder { // no-warning
  Foo<int>::Nested t1;
  Foo<char>::Nested t2;
};
