// RUN: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF
// RUN: %clang_cc1 -fsyntax-only -verify %s -DALL -Wgnu 
// RUN: %clang_cc1 -fsyntax-only -verify %s -DALL \
// RUN:   -Wgnu-alignof-expression -Wgnu-case-range -Wgnu-complex-integer -Wgnu-conditional-omitted-operand \
// RUN:   -Wgnu-empty-initializer -Wgnu-label-as-value -Wgnu-local-label -Wgnu-statement-expression
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNONE -Wgnu \
// RUN:   -Wno-gnu-alignof-expression -Wno-gnu-case-range -Wno-gnu-complex-integer -Wno-gnu-conditional-omitted-operand \
// RUN:   -Wno-gnu-empty-initializer -Wno-gnu-label-as-value -Wno-gnu-local-label -Wno-gnu-statement-expression
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF -Wgnu-alignof-expression
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DNONE -Wno-gnu-alignof-expression
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF -DCASERANGE -Wgnu-case-range
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF -DCOMPLEXINT -Wgnu-complex-integer
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF -DOMITTEDOPERAND -Wgnu-conditional-omitted-operand
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF -DEMPTYINIT -Wgnu-empty-initializer
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF -DLABELVALUE -Wgnu-label-as-value
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF -DLOCALLABEL -Wgnu-local-label
// RUNNOT: %clang_cc1 -fsyntax-only -verify %s -DALIGNOF -DSTATEMENTEXP -Wgnu-statement-expression

#if NONE
// expected-no-diagnostics
#endif


#if ALL || ALIGNOF
// expected-warning@+4 {{'_Alignof' applied to an expression is a GNU extension}}
#endif

char align;
_Static_assert(_Alignof(align) == 1, "align's alignment is wrong");


#if ALL || CASERANGE
// expected-warning@+5 {{use of GNU case range extension}}
#endif

void caserange(int x) {
  switch (x) {
  case 42 ... 44: ;
  }
}


#if ALL || COMPLEXINT
// expected-warning@+3 {{complex integer types are a GNU extension}}
#endif

_Complex short int complexint;


#if ALL || OMITTEDOPERAND
// expected-warning@+3 {{use of GNU ?: conditional expression extension, omitting middle operand}}
#endif

static const char* omittedoperand = (const char*)0 ?: "Null";


#if ALL || EMPTYINIT
// expected-warning@+3 {{use of GNU empty initializer extension}}
#endif

struct { int x; } emptyinit = {};


#if ALL || LABELVALUE
// expected-warning@+6 {{use of GNU address-of-label extension}}
// expected-warning@+7 {{use of GNU indirect-goto extension}}
#endif

void labelvalue() {
	void *ptr;
	ptr = &&foo;
foo:
	goto *ptr;
}


#if ALL || LOCALLABEL
// expected-warning@+5 {{use of GNU locally declared label extension}}
#endif

void locallabel() {
	{
		__label__ foo;
		goto foo;
foo:
		;
	}
}


#if ALL || STATEMENTEXP
// expected-warning@+5 {{use of GNU statement expression extension}}
#endif

void statementexp()
{
	int a = ({ 1; });
}
