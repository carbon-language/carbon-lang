// RUN: clang %s -verify -fsyntax-only
typedef char T[4];

T foo(int n, int m) {  }  // expected-error {{cannot return array or function}}

void foof(const char *, ...) __attribute__((__format__(__printf__, 1, 2))), barf (void);

int typedef validTypeDecl() { } // expected-error {{function definition declared 'typedef'}}

struct _zend_module_entry { }
typedef struct _zend_function_entry { } // expected-error {{cannot combine with previous 'struct' declaration specifier}}
static void buggy(int *x) { } // expected-error {{function definition declared 'typedef'}} \
                              // expected-error {{cannot combine with previous 'typedef' declaration specifier}} \
                              // expected-error {{cannot combine with previous 'struct' declaration specifier}}


