// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region --verify %s

// Test if the 'storage' region gets properly initialized after it is cast to
// 'struct sockaddr *'. 

#include <sys/socket.h>
void f(int sock) {
  struct sockaddr_storage storage;
  struct sockaddr* sockaddr = (struct sockaddr*)&storage;
  socklen_t addrlen = sizeof(storage);
  getsockname(sock, sockaddr, &addrlen);
  switch (sockaddr->sa_family) { // no-warning
  default:
    ;
  }
}
