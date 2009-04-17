// Header for PCH test stmts.c

void f0(int x) { 
  // NullStmt
  ;
  // IfStmt
  if (x) {
  } else if (x + 1) {
  }

  switch (x) {
  case 0:
    x = 17;
    break;

  case 1:
    break;

  default:
    break;
  }
}
