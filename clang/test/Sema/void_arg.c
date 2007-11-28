/* RUN: clang -fsyntax-only %s 2>&1 | grep '6 diagnostics'
 */

typedef void Void;

void foo() {
  int X;
  
  X = sizeof(int (void a));
  X = sizeof(int (int, void));
  X = sizeof(int (void, ...));

  X = sizeof(int (Void a));
  X = sizeof(int (int, Void));
  X = sizeof(int (Void, ...));

  // Accept these.
  X = sizeof(int (void));
  X = sizeof(int (Void));
}

// this is ok.
void bar(Void) {
}

