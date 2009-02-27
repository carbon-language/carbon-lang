// RUN: clang -verify %s
//
// This example was reduced from actual code in Wine 1.1.13.  GCC accepts this
// code, while the correct behavior is to reject it.
//

typedef struct _IRP {
  union {
    struct {
      union {} u; // expected-note{{previous declaration is here}}
      struct {
        union {} u; // expected-error{{error: member of anonymous struct redeclares 'u'}}
      };
    } Overlay;
  } Tail;
} IRP;
