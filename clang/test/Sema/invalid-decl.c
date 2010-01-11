// RUN: %clang_cc1 %s -fsyntax-only -verify

void test() {
    char = 4;  // expected-error {{expected identifier}}
}


// PR2400
typedef xtype (*x)(void* handle); // expected-error {{function cannot return function type}} expected-warning {{type specifier missing, defaults to 'int'}} expected-warning {{type specifier missing, defaults to 'int'}}

typedef void ytype();


typedef struct _zend_module_entry zend_module_entry;
struct _zend_module_entry {
    ytype globals_size; // expected-error {{field 'globals_size' declared as a function}}
};

zend_module_entry openssl_module_entry = {
    sizeof(zend_module_entry)
};

