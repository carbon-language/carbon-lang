//===-- string.c - String functions for the LLVM libc Library -----*- C -*-===//
// 
// A lot of this code is ripped gratuitously from glibc and libiberty.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
void *malloc(size_t);
void free(void *);

size_t strlen(const char *Str) {
  size_t Count = 0;
  while (*Str) { ++Count; ++Str; }
  return Count;
}

char *strdup(const char *str) {
  long Len = strlen(str);
  char *Result = (char*)malloc((Len+1)*sizeof(char));
  memcpy(Result, str, Len+1);
  return Result;
}

char *strcpy(char *s1, const char *s2) {
  while ((*s1++ = *s2++));
  return s1;
}

char *strcat(char *s1, const char *s2) {
  strcpy(s1+strlen(s1), s2);
  return s1;
}


/* Compare S1 and S2, returning less than, equal to or
   greater than zero if S1 is lexicographically less than,
   equal to or greater than S2.  */
int strcmp (const char *p1, const char *p2) {
  register const unsigned char *s1 = (const unsigned char *) p1;
  register const unsigned char *s2 = (const unsigned char *) p2;
  unsigned char c1, c2;

  do
    {
      c1 = (unsigned char) *s1++;
      c2 = (unsigned char) *s2++;
      if (c1 == '\0')
        return c1 - c2;
    }
  while (c1 == c2);

  return c1 - c2;
}

// http://sources.redhat.com/cgi-bin/cvsweb.cgi/libc/sysdeps/generic/?cvsroot=glibc
#if 0
typedef unsigned int op_t;
#define OPSIZ 4

void *memset (void *dstpp, int c, size_t len) {
  long long int dstp = (long long int) dstpp;

  if (len >= 8)
    {
      size_t xlen;
      op_t cccc;

      cccc = (unsigned char) c;
      cccc |= cccc << 8;
      cccc |= cccc << 16;
      if (OPSIZ > 4)
        /* Do the shift in two steps to avoid warning if long has 32 bits.  */
        cccc |= (cccc << 16) << 16;

      /* There are at least some bytes to set.
         No need to test for LEN == 0 in this alignment loop.  */
      while (dstp % OPSIZ != 0)
        {
          ((unsigned char *) dstp)[0] = c;
          dstp += 1;
          len -= 1;
        }

      /* Write 8 `op_t' per iteration until less than 8 `op_t' remain.  */
      xlen = len / (OPSIZ * 8);
      while (xlen > 0)
        {
          ((op_t *) dstp)[0] = cccc;
          ((op_t *) dstp)[1] = cccc;
          ((op_t *) dstp)[2] = cccc;
          ((op_t *) dstp)[3] = cccc;
          ((op_t *) dstp)[4] = cccc;
          ((op_t *) dstp)[5] = cccc;
          ((op_t *) dstp)[6] = cccc;
          ((op_t *) dstp)[7] = cccc;
          dstp += 8 * OPSIZ;
          xlen -= 1;
        }
      len %= OPSIZ * 8;

      /* Write 1 `op_t' per iteration until less than OPSIZ bytes remain.  */
      xlen = len / OPSIZ;
      while (xlen > 0)
        {
          ((op_t *) dstp)[0] = cccc;
          dstp += OPSIZ;
          xlen -= 1;
        }
      len %= OPSIZ;
    }

  /* Write the last few bytes.  */
  while (len > 0)
    {
      ((unsigned char *) dstp)[0] = c;
      dstp += 1;
      len -= 1;
    }

  return dstpp;
}
#endif

void *memcpy(void *dstpp, const void *srcpp, size_t len) {
  char *dstp = (char*)dstpp;
  char *srcp = (char*) srcpp;
  unsigned i;

  for (i = 0; i < len; ++i)
    dstp[i] = srcp[i];

  return dstpp;
}
