// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -add-override %t.cpp -- -I %S -std=c++11
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -add-override -override-macros %t.cpp -- -I %S -std=c++11
// RUN: FileCheck --check-prefix=MACRO --input-file=%t.cpp %s

struct A {
  virtual ~A();
  // CHECK: virtual ~A();
  void f();
  virtual void h() const;
  // CHECK: virtual void h() const;
  virtual void i() = 0;
  // CHECK: virtual void i() = 0;
};

// Test that override isn't added to non-virtual functions.
struct B : public A {
  void f();
  // CHECK: struct B
  // CHECK-NEXT: void f();
};

// Test that override is added to functions that override virtual functions.
struct C : public A {
  void h() const;
  // CHECK: struct C
  // CHECK-NEXT: void h() const override;
  // MACRO: struct C
  // MACRO-NEXT: void h() const override;
};

// Test that override isn't add to functions that overload but not override.
struct D : public A {
  void h();
  // CHECK: struct D
  // CHECK-NEXT: void h();
};

// Test that override isn't added again to functions that already have it.
struct E : public A {
  void h() const override;
  // CHECK: struct E
  // CHECK-NEXT: void h() const override;
  // MACRO: struct E
  // MACRO-NEXT: void h() const override;
};

// Test that override isn't added to the destructor.
struct F : public A {
  virtual ~F();
  // CHECK: struct F
  // CHECK-NEXT: virtual ~F();
};

// Test that override is placed before any end of line comments.
struct G : public A {
  void h() const; // comment
  void i() // comment
  {}
  // CHECK: struct G
  // CHECK-NEXT: void h() const override; // comment
  // CHECK-NEXT: void i() override // comment
  // CHECK-NEXT: {}
};

// Test that override is placed correctly if there is an inline body.
struct H : public A {
  void h() const { }
  // CHECK: struct H
  // CHECK-NEXT: void h() const override { }
};

// Test that override is placed correctly if there is a body on the next line.
struct I : public A {
  void h() const
  { }
  // CHECK: struct I
  // CHECK-NEXT: void h() const override
  // CHECK-NEXT: { }
};

// Test that override is placed correctly if there is a body outside the class.
struct J : public A {
  void h() const;
  // CHECK: struct J
  // CHECK-NEXT: void h() const override;
};

void J::h() const {
  // CHECK: void J::h() const {
}

// Test that override is placed correctly if there is a trailing return type.
struct K : public A {
  auto h() const -> void;
  // CHECK: struct K
  // CHECK-NEXT: auto h() const -> void override;
};

#define LLVM_OVERRIDE override

// Test that override isn't added if it is already specified via a macro.
struct L : public A {
  void h() const LLVM_OVERRIDE;
  // CHECK: struct L
  // CHECK-NEXT: void h() const LLVM_OVERRIDE;
  // MACRO: struct L
  // MACRO-NEXT: void h() const LLVM_OVERRIDE;
};

template <typename T>
struct M : public A {
  virtual void i();
  // CHECK: struct M
  // CHECK-NEXT: virtual void i() override;
  // MACRO: struct M
  // MACRO-NEXT: virtual void i() LLVM_OVERRIDE;
};
M<int> b;

// Test that override isn't added at the wrong place for "pure overrides"
struct APure {
  virtual APure *clone() = 0;
};
struct BPure : APure {
  virtual BPure *clone() { return new BPure(); }
};
struct CPure : BPure {
  virtual BPure *clone() = 0;
  // CHECK: struct CPure : BPure {
  // CHECK-NOT: virtual BPure *clone() = 0 override;
  // CHECK: };
};
struct DPure : CPure {
  virtual DPure *clone() { return new DPure(); }
};

// Test that override is not added on dangerous template constructs
struct Base1 {
  virtual void f();
};
struct Base2 {};
template<typename T> struct Derived : T {
  void f(); // adding 'override' here will break instantiation of Derived<Base2>
  // CHECK: struct Derived
  // CHECK-NEXT: void f();
};
Derived<Base1> d1;
Derived<Base2> d2;

#undef LLVM_OVERRIDE

struct N : public A {
  void h() const;
  // CHECK: struct N
  // CHECK-NEXT: void h() const override;
  // MACRO: struct N
  // MACRO-NEXT: void h() const override;
};
