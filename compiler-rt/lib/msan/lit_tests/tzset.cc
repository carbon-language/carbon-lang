// RUN: %clangxx_msan -m64 -O0 %s -o %t && %t

#include <stdlib.h>
#include <string.h>
#include <time.h>

extern char *tzname[2];

int main(void) {
  if (!strlen(tzname[0]) || !strlen(tzname[1]))
    exit(1);
  tzset();
  if (!strlen(tzname[0]) || !strlen(tzname[1]))
    exit(1);
  return 0;
}
