extern int X;
template <class T> void bar() {
  if (X) {
    X *= 4;
  }
}
void a();
void b();
