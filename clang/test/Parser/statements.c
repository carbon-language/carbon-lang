// RUN: %clang_cc1 -fsyntax-only -verify %s

void test1() {
  { ; {  ;;}} ;;
}

void test2() {
  if (0) { if (1) {} } else { }

  do { } while (0); 
  
  while (0) while(0) do ; while(0);

  for ((void)0;0;(void)0)
    for (;;)
      for ((void)9;0;(void)2)
        ;
  for (int X = 0; 0; (void)0);
}

void test3() {
    switch (0) {
    
    case 4:
      if (0) {
    case 6: ;
      }
    default:
      ;     
  }
}

void test4() {
  if (0);  // expected-warning {{if statement has empty body}}
  
  int X;  // declaration in a block.
  
foo:  if (0); // expected-warning {{if statement has empty body}}
}

typedef int t;
void test5() {
  if (0);   // expected-warning {{if statement has empty body}}

  t x = 0;

  if (0);  // expected-warning {{if statement has empty body}}
}


void test6(void) { 
  do 
    .           // expected-error {{expected expression}}
   while (0);
}

int test7() {
  return 4     // expected-error {{expected ';' after return statement}}
}
