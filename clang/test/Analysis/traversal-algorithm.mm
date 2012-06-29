// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpTraversal -std=c++11 %s | FileCheck -check-prefix=DFS %s

int a();
int b();
int c();

int work();

void test(id input) {
  if (a()) {
    if (a())
      b();
    else
      c();
  } else {
    if (b())
      a();
    else
      c();
  }

  if (a())
    work();
}

// This ordering assumes that true cases happen before the false cases.

// BFS: 10 IfStmt
// BFS-NEXT: 11 IfStmt
// BFS-NEXT: 16 IfStmt
// BFS-NEXT: 22 IfStmt
// BFS-NEXT: 22 IfStmt
// BFS-NEXT: 22 IfStmt
// BFS-NEXT: 22 IfStmt
// BFS-NEXT: --END PATH--
// BFS-NEXT: --END PATH--
// BFS-NEXT: --END PATH--
// BFS-NEXT: --END PATH--
// BFS-NEXT: --END PATH--
// BFS-NEXT: --END PATH--
// BFS-NEXT: --END PATH--
// BFS-NEXT: --END PATH--

// And this ordering assumes that false cases happen before the true cases.

// DFS: 10 IfStmt
// DFS-NEXT: 16 IfStmt
// DFS-NEXT: 22 IfStmt
// DFS-NEXT: --END PATH--
// DFS-NEXT: --END PATH--
// DFS-NEXT: 22 IfStmt
// DFS-NEXT: --END PATH--
// DFS-NEXT: --END PATH--
// DFS-NEXT: 11 IfStmt
// DFS-NEXT: 22 IfStmt
// DFS-NEXT: --END PATH--
// DFS-NEXT: --END PATH--
// DFS-NEXT: 22 IfStmt
// DFS-NEXT: --END PATH--
// DFS-NEXT: --END PATH--


void testLoops(id input) {
  while (a()) {
    work();
    work();
    work();
  }

  for (int i = 0; i != b(); ++i) {
    work();
  }

  for (id x in input) {
    work();
    work();
    work();
  }

  int z[] = {1,2,3};
  for (int y : z) {
    work();
    work();
    work();
  }
}

// BFS: 64 WhileStmt
// BFS: 70 ForStmt
// BFS-NOT-NEXT: ObjCForCollectionStmt
// BFS: 74 ObjCForCollectionStmt
// BFS: 81 CXXForRangeStmt

// DFS: 64 While
// DFS-NEXT: 70 ForStmt
// DFS-NEXT: 74 ObjCForCollectionStmt
// DFS-NEXT: 81 CXXForRangeStmt
