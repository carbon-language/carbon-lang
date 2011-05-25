// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -disable-free -analyzer-eagerly-assume -analyzer-checker=core -analyzer-checker=deadcode -verify %s

unsigned long strlen(const char *);

int size_rdar9373039 = 1;
int rdar9373039() {
  int x;
  int j = 0;

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    x = 1;
    
  // strlen doesn't invalidate the value of 'size_rdar9373039'.
  int extra = (2 + strlen ("Clang") + ((4 - ((unsigned int) (2 + strlen ("Clang")) % 4)) % 4)) + (2 + strlen ("1.0") + ((4 - ((unsigned int) (2 + strlen ("1.0")) % 4)) % 4));

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    j += x; // no-warning

  return j;
}

int foo_rdar9373039(const char *);

int rdar93730392() {
  int x;
  int j = 0;

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    x = 1;
    
  int extra = (2 + foo_rdar9373039 ("Clang") + ((4 - ((unsigned int) (2 + foo_rdar9373039 ("Clang")) % 4)) % 4)) + (2 + foo_rdar9373039 ("1.0") + ((4 - ((unsigned int) (2 + foo_rdar9373039 ("1.0")) % 4)) % 4)); // expected-warning {{never read}}

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    j += x; // expected-warning {{garbage}}

  return j;
}

