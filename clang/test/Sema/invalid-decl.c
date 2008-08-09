// RUN: clang %s -fsyntax-only -verify

void test() {
    char = 4;  // expected-error {{expected identifier}} expected-error{{declarator requires an identifier}}

}


// PR2400
typedef xtype (*zend_stream_fsizer_t)(void* handle); // expected-error {{function cannot return array or function type}}

typedef struct _zend_module_entry zend_module_entry;
struct _zend_module_entry {
    xtype globals_size; // expected-error {{field 'globals_size' declared as a function}}
};

zend_module_entry openssl_module_entry = {
    sizeof(zend_module_entry)
};

