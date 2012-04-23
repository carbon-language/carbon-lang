// RUN: %clang_cc1 %s -verify -fsyntax-only
// rdar: // 6734520

typedef int INT1 __attribute__((deprecated("Please avoid INT1")));

typedef INT1 INT2 __attribute__ ((__deprecated__("Please avoid INT2")));

typedef INT1 INT1a; // expected-warning {{'INT1' is deprecated: Please avoid INT1}}

typedef INT1 INT1b __attribute__ ((deprecated("Please avoid INT1b")));

INT1 should_be_unavailable; // expected-warning {{'INT1' is deprecated: Please avoid INT1}}
INT1a should_not_be_deprecated;

INT1 f1(void) __attribute__ ((deprecated("Please avoid f1"))); 
INT1 f2(void); // expected-warning {{'INT1' is deprecated: Please avoid INT1}}

typedef enum {red, green, blue} Color __attribute__((deprecated("Please avoid Color")));
 

Color c1; // expected-warning {{'Color' is deprecated: Please avoid Color}}

int g1;
int g2 __attribute__ ((deprecated("Please avoid g2"))); // expected-note {{'g2' declared here}}

int func1()
{
   int (*pf)() = f1; // expected-warning {{'f1' is deprecated: Please avoid f1}}
   int i = f2();
   return g1 + g2; // expected-warning {{'g2' is deprecated: Please avoid g2}}
}
