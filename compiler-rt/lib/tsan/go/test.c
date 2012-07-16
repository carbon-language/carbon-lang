#include <stdio.h>

void __tsan_init();
void __tsan_fini();

int goCallbackCommentPc(void *pc, char **img, char **rtn, char **file, int *l) {
  return 0;
}

int main(void) {
  __tsan_init();
  printf("OK\n");
  __tsan_fini();
  return 0;
}
