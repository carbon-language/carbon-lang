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
    switch (x >> 1) {
    case 7:
      // fall through
    case 9:
      break;
    }
    x += 2;
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

  for (int y = x; y < 20; ++y) {
    if (x + y == 12)
      return;
  }

  int z = x, *y, j = 5;
}

int f1(int x) {
  switch (x) {
  case 17:
    return 12;

  default:
    break;
  }

  return x*2;
}

const char* what_is_my_name(void) { return __func__; }
