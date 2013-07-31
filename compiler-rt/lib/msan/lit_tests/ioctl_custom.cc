// RUN: %clangxx_msan -m64 -O0 -g %s -o %t && %t
// RUN: %clangxx_msan -m64 -O3 -g %s -o %t && %t

// RUN: %clangxx_msan -DPOSITIVE -m64 -O0 -g %s -o %t && not %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -DPOSITIVE -m64 -O3 -g %s -o %t && not %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <net/if.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char **argv) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);

  struct ifreq ifreqs[20];
  struct ifconf ifc;
  ifc.ifc_ifcu.ifcu_req = ifreqs;
#ifndef POSITIVE
  ifc.ifc_len = sizeof(ifreqs);
#endif
  int res = ioctl(fd, SIOCGIFCONF, (void *)&ifc);
  // CHECK: UMR in ioctl{{.*}} at offset 0
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: #{{.*}} in main {{.*}}ioctl_custom.cc:[[@LINE-3]]
  assert(res == 0);
  for (int i = 0; i < ifc.ifc_len / sizeof(*ifc.ifc_ifcu.ifcu_req); ++i)
    printf("%d  %zu  %s\n", i, strlen(ifreqs[i].ifr_name), ifreqs[i].ifr_name);
  return 0;
}
