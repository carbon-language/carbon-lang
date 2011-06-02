// RUN: %llvmgcc -S %s -o - | FileCheck %s
// <rdar://problem/9402870>
typedef struct __TestResult TestResult;
typedef struct __TestResult* TestResultRef;

typedef struct __TestImplement TestImplement;
typedef struct __TestImplement* TestImplementRef;

typedef char*(*TestNameFunction)(void*);
typedef void(*TestRunFunction)(void*,TestResult*);
typedef int(*TestCountTestCasesFunction)(void*);

struct __TestImplement {
    TestNameFunction name;
    TestRunFunction run;
    TestCountTestCasesFunction countTestCases;
};

typedef struct __Test Test;
typedef struct __Test* TestRef;

struct __Test {
    TestImplement* isa;
};

typedef struct __TestCase TestCase;
typedef struct __TestCase* TestCaseRef;

struct __TestCase {
    TestImplement* isa;
    const char *name;
    void(*setUp)(void);
    void(*tearDown)(void);
    void(*runTest)(void);
};

extern const TestImplement TestCaseImplement;

typedef struct __TestFixture TestFixture;
typedef struct __TestFixture* TestFixtureRef;

struct __TestFixture {
    const char *name;
    void(*test)(void);
};

typedef struct __TestCaller TestCaller;
typedef struct __TestCaller* TestCallerRef;

struct __TestCaller {
    TestImplement* isa;
    const char *name;
    void(*setUp)(void);
    void(*tearDown)(void);
    int numberOfFixtuers;
    TestFixture *fixtuers;
};

extern const TestImplement TestCallerImplement;

void PassToFunction(const TestImplement*);

const char* TestCaller_name(TestCaller* self) {
  return self->name;
}

void TestCaller_run(TestCaller* self,TestResult* result) {
  TestCase cs = { (TestImplement*)&TestCaseImplement, 0, 0, 0, 0, };
  int i;
  cs.setUp = self->setUp;
  cs.tearDown = self->tearDown;
  for (i=0; i<self->numberOfFixtuers; i++) {
    cs.name = self->fixtuers[i].name;
    cs.runTest = self->fixtuers[i].test;
    ((Test*)(void *)&cs)->isa->run((void *)&cs,result);
  }
}

int TestCaller_countTestCases(TestCaller* self) {
  PassToFunction(&TestCallerImplement);
  return self->numberOfFixtuers;
}

// CHECK: @C.0.1526 = internal unnamed_addr constant
// CHECK-NOT: @TestCaseImplement = external unnamed_addr constant %struct.TestImplement
// CHECK: @TestCaseImplement = external constant %struct.TestImplement
const TestImplement TestCallerImplement = {
  (TestNameFunction)TestCaller_name,
  (TestRunFunction)TestCaller_run,
  (TestCountTestCasesFunction)TestCaller_countTestCases,
};
