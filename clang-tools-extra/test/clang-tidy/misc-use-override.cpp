// RUN: $(dirname %s)/check_clang_tidy.sh %s misc-use-override %t
// REQUIRES: shell

#define ABSTRACT = 0

#define OVERRIDE override
#define VIRTUAL virtual
#define NOT_VIRTUAL
#define NOT_OVERRIDE

#define MUST_USE_RESULT __attribute__((warn_unused_result))
#define UNUSED __attribute__((unused))

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
  virtual bool l() MUST_USE_RESULT UNUSED;
  virtual bool n() MUST_USE_RESULT UNUSED;

  virtual void m();
  virtual void o() __attribute__((unused));
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: Prefer using 'override' or (rarely) 'final' instead of 'virtual'
  // CHECK-FIXES: {{^}}  ~SimpleCases() override;

  void a();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: Annotate this
  // CHECK-FIXES: {{^}}  void a() override;

  void b() override;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  void b() override;

  virtual void c();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void c() override;

  virtual void d() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  void d() override;

  virtual void e() = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void e() override = 0;

  virtual void f()=0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void f()override =0;

  virtual void g() ABSTRACT;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void g() override ABSTRACT;

  virtual void j() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void j() const override;

  virtual MustUseResultObject k();  // Has an implicit attribute.
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: Prefer using
  // CHECK-FIXES: {{^}}  MustUseResultObject k() override;

  virtual bool l() MUST_USE_RESULT UNUSED;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  bool l() override MUST_USE_RESULT UNUSED;

  virtual bool n() UNUSED MUST_USE_RESULT;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  bool n() override UNUSED MUST_USE_RESULT;

  virtual void m() override final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  void m() final;

  virtual void o() __attribute__((unused));
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void o() override __attribute__((unused));
};

// CHECK-MESSAGES-NOT: warning:

void SimpleCases::c() {}
// CHECK-FIXES: {{^}}void SimpleCases::c() {}

SimpleCases::~SimpleCases() {}
// CHECK-FIXES: {{^}}SimpleCases::~SimpleCases() {}

struct DefaultedDestructor : public Base {
  DefaultedDestructor() {}
  virtual ~DefaultedDestructor() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: Prefer using
  // CHECK-FIXES: {{^}}  ~DefaultedDestructor() override = default;
};

struct FinalSpecified : public Base {
public:
  virtual ~FinalSpecified() final;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: Annotate this
  // CHECK-FIXES: {{^}}  ~FinalSpecified() final;

  void b() final;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  void b() final;

  virtual void d() final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  void d() final;

  virtual void e() final = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  void e() final = 0;

  virtual void j() const final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  void j() const final;

  virtual bool l() final MUST_USE_RESULT UNUSED;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  bool l() final MUST_USE_RESULT UNUSED;
};

struct InlineDefinitions : public Base {
public:
  virtual ~InlineDefinitions() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: Prefer using
  // CHECK-FIXES: {{^}}  ~InlineDefinitions() override {}

  void a() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: Annotate this
  // CHECK-FIXES: {{^}}  void a() override {}

  void b() override {}
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  void b() override {}

  virtual void c() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void c() override {}

  virtual void d() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  void d() override {}

  virtual void j() const {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void j() const override {}

  virtual MustUseResultObject k() {}  // Has an implicit attribute.
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: Prefer using
  // CHECK-FIXES: {{^}}  MustUseResultObject k() override {}

  virtual bool l() MUST_USE_RESULT UNUSED {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  bool l() override MUST_USE_RESULT UNUSED {}
};

struct Macros : public Base {
  // Tests for 'virtual' and 'override' being defined through macros. Basically
  // give up for now.
  NOT_VIRTUAL void a() NOT_OVERRIDE;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: Annotate this
  // CHECK-FIXES: {{^}}  NOT_VIRTUAL void a() override NOT_OVERRIDE;

  VIRTUAL void b() NOT_OVERRIDE;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  VIRTUAL void b() override NOT_OVERRIDE;

  NOT_VIRTUAL void c() OVERRIDE;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  NOT_VIRTUAL void c() OVERRIDE;

  VIRTUAL void d() OVERRIDE;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  VIRTUAL void d() OVERRIDE;

#define FUNC(return_type, name) return_type name()
  FUNC(void, e);
  // CHECK-FIXES: {{^}}  FUNC(void, e);

#define F virtual void f();
  F
  // CHECK-FIXES: {{^}}  F

  VIRTUAL void g() OVERRIDE final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Annotate this
  // CHECK-FIXES: {{^}}  VIRTUAL void g() final;
};

// Tests for templates.
template <typename T> struct TemplateBase {
  virtual void f(T t);
};

template <typename T> struct DerivedFromTemplate : public TemplateBase<T> {
  virtual void f(T t);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Prefer using
  // CHECK-FIXES: {{^}}  void f(T t) override;
};
void f() { DerivedFromTemplate<int>().f(2); }

template <class C>
struct UnusedMemberInstantiation : public C {
  virtual ~UnusedMemberInstantiation() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: Prefer using
  // CHECK-FIXES: {{^}}  ~UnusedMemberInstantiation() override {}
};
struct IntantiateWithoutUse : public UnusedMemberInstantiation<Base> {};

struct Base2 {
  virtual ~Base2() {}
  virtual void a();
};

// The OverrideAttr isn't propagated to specializations in all cases. Make sure
// we don't add "override" a second time.
template <int I>
struct MembersOfSpecializations : public Base2 {
  void a() override;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  void a() override;
};
template <> void MembersOfSpecializations<3>::a() {}
void ff() { MembersOfSpecializations<3>().a(); };
