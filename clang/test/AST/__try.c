// RUN: %ast_test -fborland-extensions %s

#define JOIN2(x,y) x ## y
#define JOIN(x,y) JOIN2(x,y)
#define TEST2(name) JOIN(name,__LINE__)
#define TEST TEST2(test)
typedef int DWORD;

DWORD FilterExpression();

void TEST() {
  __try // expected-stmt-class-name{{SEHTryStmt}}
  { // expected-stmt-class-name{{CompoundStmt}}
  }
  __except ( FilterExpression() ) // expected-stmt-class-name{{SEHExceptStmt}} expected-stmt-class-name{{CallExpr}} \
    // expected-expr-type{{DWORD}}
  { // expected-stmt-class-name{{CompoundStmt}}
  }
}

void TEST() {
  __try // expected-stmt-class-name{{SEHTryStmt}}
  { // expected-stmt-class-name{{CompoundStmt}}
  }
  __finally // expected-stmt-class-name{{SEHFinallyStmt}}
  { // expected-stmt-class-name{{CompoundStmt}}
  }
}
