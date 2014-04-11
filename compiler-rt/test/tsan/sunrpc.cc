// RUN: %clang_tsan -O1 %s -o %t && %t

#include <pthread.h>
#include <rpc/xdr.h>

void *thr(void *p) {
  XDR xdrs;
  char buf[100];
  xdrmem_create(&xdrs, buf, sizeof(buf), XDR_ENCODE);
  xdr_destroy(&xdrs);
  return 0;
}

int main(int argc, char *argv[]) {
  pthread_t th[2];
  pthread_create(&th[0], 0, thr, 0);
  pthread_create(&th[1], 0, thr, 0);
  pthread_join(th[0], 0);
  pthread_join(th[1], 0);
  return 0;
}
