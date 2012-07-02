// Note: the run lines follow their respective tests, since line/column
// matter in this test.

/// Aaa.
void T1(float x, float y);

/// Bbb.
class T2 {
public:
  /// Ccc.
  void T3();

  int T4; ///< Ddd.
};

/// Eee.
namespace T5 {
}

void test() {

  T2 t2;
  t2.
}

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:21:1 %s | FileCheck -check-prefix=CC1 %s
// CHECK-CC1: FunctionDecl:{ResultType void}{TypedText T1}{{.*}}(brief comment: Aaa.)
// CHECK-CC1: ClassDecl:{TypedText T2}{{.*}}(brief comment: Bbb.)
// CHECK-CC1: Namespace:{TypedText T5}{{.*}}(brief comment: Eee.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:23:6 %s | FileCheck -check-prefix=CC2 %s
// CHECK-CC2: CXXMethod:{ResultType void}{TypedText T3}{{.*}}(brief comment: Ccc.)
// CHECK-CC2: FieldDecl:{ResultType int}{TypedText T4}{{.*}}(brief comment: Ddd.)
