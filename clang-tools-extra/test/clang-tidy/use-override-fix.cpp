// RUN: $(dirname %s)/check_clang_tidy_fix.sh %s misc-use-override %t
// REQUIRES: shell

#define ABSTRACT = 0

#define OVERRIDE override
#define VIRTUAL virtual
#define NOT_VIRTUAL
#define NOT_OVERRIDE

#define MUST_USE_RESULT __attribute__((warn_unused_result))

struct MUST_USE_RESULT MustUseResultObject {};

struct Base {
  virtual ~Base() {}
  virtual void a();
  virtual void b();
  virtual void c();
  virtual void d();
  virtual void e() = 0;
  virtual void f() = 0;
  virtual void g() = 0;

  virtual void j() const;
  virtual MustUseResultObject k();
  virtual bool l() MUST_USE_RESULT;

  virtual void m();
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases();
  // CHECK: {{^  ~SimpleCases\(\) override;}}

  void a();
  // CHECK: {{^  void a\(\) override;}}
  void b() override;
  // CHECK: {{^  void b\(\) override;}}
  virtual void c();
  // CHECK: {{^  void c\(\) override;}}
  virtual void d() override;
  // CHECK: {{^  void d\(\) override;}}

  virtual void e() = 0;
  // CHECK: {{^  void e\(\) override = 0;}}
  virtual void f()=0;
  // CHECK: {{^  void f\(\)override =0;}}
  virtual void g() ABSTRACT;
  // CHECK: {{^  void g\(\) override ABSTRACT;}}

  virtual void j() const;
  // CHECK: {{^  void j\(\) const override;}}
  virtual MustUseResultObject k();  // Has an implicit attribute.
  // CHECK: {{^  MustUseResultObject k\(\) override;}}
  virtual bool l() MUST_USE_RESULT; // Has an explicit attribute
  // CHECK: {{^  bool l\(\) override MUST_USE_RESULT;}}

  virtual void m() override final;
  // CHECK: {{^  void m\(\) final;}}
};

void SimpleCases::i() {}
// CHECK: {{^void SimpleCases::i\(\) {}}}

SimpleCases::~SimpleCases() {}
// CHECK: {{^SimpleCases::~SimpleCases\(\) {}}}

struct DefaultedDestructor : public Base {
  DefaultedDestructor() {}
  virtual ~DefaultedDestructor() = default;
  // CHECK: {{^  ~DefaultedDestructor\(\) override = default;}}
};

struct FinalSpecified : public Base {
public:
  virtual ~FinalSpecified() final;
  // CHECK: {{^  ~FinalSpecified\(\) final;}}

  void b() final;
  // CHECK: {{^  void b\(\) final;}}
  virtual void d() final;
  // CHECK: {{^  void d\(\) final;}}

  virtual void e() final = 0;
  // CHECK: {{^  void e\(\) final = 0;}}

  virtual void j() const final;
  // CHECK: {{^  void j\(\) const final;}}
  virtual bool l() final MUST_USE_RESULT;
  // CHECK: {{^  bool l\(\) final MUST_USE_RESULT;}}
};

struct InlineDefinitions : public Base {
public:
  virtual ~InlineDefinitions() {}
  // CHECK: {{^  ~InlineDefinitions\(\) override {}}}

  void a() {}
  // CHECK: {{^  void a\(\) override {}}}
  void b() override {}
  // CHECK: {{^  void b\(\) override {}}}
  virtual void c() {}
  // CHECK: {{^  void c\(\) override {}}}
  virtual void d() override {}
  // CHECK: {{^  void d\(\) override {}}}

  virtual void j() const {}
  // CHECK: {{^  void j\(\) const override {}}}
  virtual MustUseResultObject k() {}  // Has an implicit attribute.
  // CHECK: {{^  MustUseResultObject k\(\) override {}}}
  virtual bool l() MUST_USE_RESULT {} // Has an explicit attribute
  // CHECK: {{^  bool l\(\) override MUST_USE_RESULT {}}}
};

struct Macros : public Base {
  // Tests for 'virtual' and 'override' being defined through macros. Basically
  // give up for now.
  NOT_VIRTUAL void a() NOT_OVERRIDE;
  // CHECK: {{^  NOT_VIRTUAL void a\(\) override NOT_OVERRIDE;}}

  VIRTUAL void b() NOT_OVERRIDE;
  // CHECK: {{^  VIRTUAL void b\(\) override NOT_OVERRIDE;}}

  NOT_VIRTUAL void c() OVERRIDE;
  // CHECK: {{^  NOT_VIRTUAL void c\(\) OVERRIDE;}}

  VIRTUAL void d() OVERRIDE;
  // CHECK: {{^  VIRTUAL void d\(\) OVERRIDE;}}

#define FUNC(name, return_type) return_type name()
  FUNC(void, e);
  // CHECK: {{^  FUNC\(void, e\);}}

#define F virtual void f();
  F
  // CHECK: {{^  F}}

  VIRTUAL void g() OVERRIDE final;
  // CHECK: {{^  VIRTUAL void g\(\) final;}}
};

// Tests for templates.
template <typename T> struct TemplateBase {
  virtual void f(T t);
};

template <typename T> struct DerivedFromTemplate : public TemplateBase<T> {
  virtual void f(T t);
  // CHECK: {{^  void f\(T t\) override;}}
};
void f() { DerivedFromTemplate<int>().f(2); }

template <class C>
struct UnusedMemberInstantiation : public C {
  virtual ~UnusedMemberInstantiation() {}
  // CHECK: {{^  ~UnusedMemberInstantiation\(\) override {}}}
};
struct IntantiateWithoutUse : public UnusedMemberInstantiation<Base> {};

// The OverrideAttr isn't propagated to specializations in all cases. Make sure
// we don't add "override" a second time.
template <int I>
struct MembersOfSpecializations : public Base {
  void a() override;
  // CHECK: {{^  void a\(\) override;}}
};
template <> void MembersOfSpecializations<3>::a() {}
void f() { MembersOfSpecializations<3>().a(); };
