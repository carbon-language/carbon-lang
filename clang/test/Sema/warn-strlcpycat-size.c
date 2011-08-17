// RUN: %clang_cc1 -Wstrlcpy-size -verify -fsyntax-only %s

typedef unsigned long size_t;
size_t strlcpy (char * restrict dst, const char * restrict src, size_t size);
size_t strlcat (char * restrict dst, const char * restrict src, size_t size);
size_t strlen (const char *s);

char s1[100];
char s2[200];
char * s3;

struct {
  char f1[100];
  char f2[100][3];
} s4, **s5;

int x;

void f(void)
{
  strlcpy(s1, s2, sizeof(s1)); // no warning
  strlcpy(s1, s2, sizeof(s2)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcpy(s1, s3, strlen(s3)+1); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcat(s2, s3, sizeof(s3)); // expected-warning {{size argument in 'strlcat' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcpy(s4.f1, s2, sizeof(s2)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcpy((*s5)->f2[x], s2, sizeof(s2)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcpy(s1+3, s2, sizeof(s2)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}}
}
