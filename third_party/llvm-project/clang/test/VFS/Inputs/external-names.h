void foo(char **c) {
  *c = __FILE__;
  int x = c; // produce a diagnostic
}
