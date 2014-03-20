extern int i;
extern long long x __attribute__((weak));
void foo();

int main() {
  foo();
  return i + x;
}
