// RUN: %clangxx -O0 -g %s -lcrypt -o %t && %run %t

// crypt.h is missing from Android.
// UNSUPPORTED: android

#include <assert.h>
#include <unistd.h>
#include <cstring>
#include <crypt.h>

int main(int argc, char **argv) {
  {
    crypt_data cd;
    cd.initialized = 0;
    char *p = crypt_r("abcdef", "xz", &cd);
    volatile size_t z = strlen(p);
  }
  {
    crypt_data cd;
    cd.initialized = 0;
    char *p = crypt_r("abcdef", "$1$", &cd);
    volatile size_t z = strlen(p);
  }
  {
    crypt_data cd;
    cd.initialized = 0;
    char *p = crypt_r("abcdef", "$5$", &cd);
    volatile size_t z = strlen(p);
  }
  {
    crypt_data cd;
    cd.initialized = 0;
    char *p = crypt_r("abcdef", "$6$", &cd);
    volatile size_t z = strlen(p);
  }
}
