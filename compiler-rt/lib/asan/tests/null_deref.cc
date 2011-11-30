__attribute__((noinline))
static void NullDeref(int *ptr) {
  ptr[10]++;
}
int main() {
  NullDeref((int*)0);
}
