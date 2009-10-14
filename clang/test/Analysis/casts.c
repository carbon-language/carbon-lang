// RUN: clang-cc -triple x86_64-apple-darwin9 -analyze -checker-cfref -analyzer-store=region --verify %s

// Test if the 'storage' region gets properly initialized after it is cast to
// 'struct sockaddr *'. 

typedef unsigned char __uint8_t;
typedef unsigned int __uint32_t;
typedef __uint32_t __darwin_socklen_t;
typedef __uint8_t sa_family_t;
typedef __darwin_socklen_t socklen_t;
struct sockaddr { sa_family_t sa_family; };
struct sockaddr_storage {};

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

struct s {
  struct s *value;
};

void f1(struct s **pval) {
  int *tbool = ((void*)0);
  struct s *t = *pval;
  pval = &(t->value);
  tbool = (int *)pval; // use the cast-to type 'int *' to create element region.
  char c = (unsigned char) *tbool; // Should use cast-to type to create symbol.
  if (*tbool == -1) // here load the element region with the correct type 'int'
    (void)3;
}

void f2(const char *str) {
 unsigned char ch, cl, *p;

 p = (unsigned char *)str;
 ch = *p++; // use cast-to type 'unsigned char' to create element region.
 cl = *p++;
 if(!cl)
    cl = 'a';
}
