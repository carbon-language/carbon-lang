// clang -target mipsel-linux-gnu -shared -fPIC -lc dynamic-table.c \
//       -o dynamic-table-so.mips
int puts(const char *);

void foo(void) {
  puts("Hello, World");
}
