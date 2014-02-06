// RUN: %clangxx_msan -m64 -O0 %s -o %t && %t %p 2>&1
// RUN: %clangxx_msan -m64 -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %t %p 2>&1
// RUN: %clangxx_msan -m64 -O3 %s -o %t && %t %p 2>&1

#include <assert.h>
#include <errno.h>
#include <ifaddrs.h>
#include <stdio.h>

#include <vector>

#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  struct ifaddrs *ifas;

  assert(0 == __msan_test_shadow(&ifas, sizeof(ifaddrs *)));
  int res = getifaddrs(&ifas);
  if (res == -1) {
    assert(errno == ENOSYS);
    printf("getifaddrs() is not implemented\n");
    return 0;
  }
  assert(res == 0);
  assert(-1 == __msan_test_shadow(&ifas, sizeof(ifaddrs *)));

  std::vector<ifaddrs *> ifas_vector;
  ifaddrs *p = ifas;
  while (p) {
    ifas_vector.push_back(p);
    assert(-1 == __msan_test_shadow(p, sizeof(ifaddrs)));
    p = p->ifa_next;
  }

  freeifaddrs(ifas);
  for (int i = 0; i < ifas_vector.size(); i++) {
    ifaddrs *p = ifas_vector[i];
    assert(0 == __msan_test_shadow(p, sizeof(ifaddrs)));
  }
  return 0;
}
