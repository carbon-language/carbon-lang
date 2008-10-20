// RUN: clang -fsyntax-only -verify %s
void donotwarn();

int (^IFP) ();
int (^II) (int);
int test1() {
	int (^PFR) (int) = 0;	// OK
	PFR = II;	// OK

	if (PFR == II)	// OK
	  donotwarn();

	if (PFR == IFP) // expected-error {{comparison of distinct block types}}
	  donotwarn();

	if (PFR == (int (^) (int))IFP) // OK
	  donotwarn();

	if (PFR == 0) // OK
	  donotwarn();

	if (PFR)	// OK
	  donotwarn();

	if (!PFR)	// OK
	  donotwarn();

	return PFR != IFP;	// expected-error {{comparison of distinct block types}}
}

int test2(double (^S)()) {
   double (^I)(int)  = (void*) S;
   (void*)I = (void *)S; 	// expected-error {{expression is not assignable}}

   void *pv = I;

   pv = S;		

   I(1);
 
   return (void*)I == (void *)S;
}

int^ x; // expected-error {{block pointer to non-function type is invalid}}
int^^ x1; // expected-error {{block pointer to non-function type is invalid}}

int test3() {
	char *^ y; // expected-error {{block pointer to non-function type is invalid}}
}



enum {NSBIRLazilyAllocated = 0};

int test4(int argc) {  // rdar://6251437
  ^{
    switch (argc) {
      case NSBIRLazilyAllocated:  // is an integer constant expression.
      default:
        break;
    }
  }();
  return 0;
}


// rdar://6257721 - reference to static/global is byref by default.
static int test5g;
void test5() {
  bar(^{ test5g = 1; });
}

