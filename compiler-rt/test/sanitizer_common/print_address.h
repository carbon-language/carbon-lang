#include <stdio.h>
#include <stdarg.h>

void print_address(const char *str, int n, ...) {
  fprintf(stderr, "%s", str);
  va_list ap;
  va_start(ap, n);
  while (n--) {
    void *p = va_arg(ap, void *);
#if defined(__x86_64__) || defined(__aarch64__) || defined(__powerpc64__) || \
    defined(__s390x__) || (defined(__riscv) && __riscv_xlen == 64)
    // On FreeBSD, the %p conversion specifier works as 0x%x and thus does not
    // match to the format used in the diagnotic message.
    fprintf(stderr, "0x%012lx ", (unsigned long) p);
#elif defined(__i386__) || defined(__arm__)
    fprintf(stderr, "0x%08lx ", (unsigned long) p);
#elif defined(__mips64)
    fprintf(stderr, "0x%010lx ", (unsigned long) p);
#endif
  }
  fprintf(stderr, "\n");
}
