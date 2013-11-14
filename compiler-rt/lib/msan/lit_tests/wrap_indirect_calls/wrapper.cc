int f_replaced(int x, int y);
int f_replacement(int x, int y);

int cnt;

extern "C" void *wrapper(void *p) {
  ++cnt;
  if (p == (void *)f_replaced)
    return (void *)f_replacement;
  return p;
}
