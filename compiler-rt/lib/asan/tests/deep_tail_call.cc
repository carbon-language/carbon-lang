int global[10];
__attribute__((noinline))
void call4(int i) { global[i+10]++; }
__attribute__((noinline))
void call3(int i) { call4(i); }
__attribute__((noinline))
void call2(int i) { call3(i); }
__attribute__((noinline))
void call1(int i) { call2(i); }
int main(int argc, char **argv) {
  call1(argc);
  return global[0];
}
