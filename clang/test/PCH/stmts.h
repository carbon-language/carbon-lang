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

  while (x > 20) {
    if (x > 30) {
      --x;
      continue;
    } else if (x < 5)
      break;
  }

  do {
    x++;
  } while (x < 10);

  for (; x < 20; ++x) ;
}
