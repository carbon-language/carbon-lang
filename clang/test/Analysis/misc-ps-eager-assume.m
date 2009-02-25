// RUN: clang -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=range --verify -fblocks %s -analyzer-eagerly-assume

void handle_assign_of_condition(int x) {
  // The cast to 'short' causes us to lose symbolic constraint.
  short y = (x != 0);
  char *p = 0;
  if (y) {
    // This should be infeasible.
    if (!(x != 0)) {
      *p = 1;  // no-warning
    }
  }
}

