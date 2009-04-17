// Test this without pch.
// RUN: clang-cc -fblocks -include %S/stmts.h -fsyntax-only -ast-print -o - %s

// Test with pch.
// RUN: clang-cc -emit-pch -fblocks -o %t %S/stmts.h &&
// RUN: clang-cc -fblocks -include-pch %t -fsyntax-only -ast-print -o - %s 

void g0(void) { f0(5); }
int g1(int x) { return f1(x); }
const char* query_name(void) { return what_is_my_name(); }

int use_computed_goto(int x) { return computed_goto(x); }

int get_weird_max(int x, int y) { return weird_max(x, y); }
