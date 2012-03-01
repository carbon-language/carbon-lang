// RUN: %clang_cc1 %s -fsyntax-only -verify 
// rdar://10961370

typedef struct __CFError * CFErrorRef; // expected-note {{forward declaration of 'struct __CFError'}}

void junk(int, ...);

int main()
{
 CFErrorRef error;
 junk(1, *error); // expected-error {{incomplete type 'struct __CFError' where a complete type is required}}
}
