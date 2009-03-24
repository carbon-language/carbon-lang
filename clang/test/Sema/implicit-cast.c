// RUN: clang-cc -fsyntax-only %s

static char *test1(int cf) {
  return cf ? "abc" : 0;
}
static char *test2(int cf) {
  return cf ? 0 : "abc";
}
