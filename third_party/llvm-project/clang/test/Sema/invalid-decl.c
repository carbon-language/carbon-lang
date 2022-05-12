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

// <rdar://problem/11067144>
typedef int (FunctionType)(int *value);
typedef struct {
  UndefinedType undef; // expected-error {{unknown type name 'UndefinedType'}}
  FunctionType fun; // expected-error {{field 'fun' declared as a function}}
} StructType;
void f(StructType *buf) {
  buf->fun = 0;
}

// rdar://11743706
static void bar(hid_t, char); // expected-error {{expected identifier}}

static void bar(hid_t p, char); // expected-error {{unknown type name 'hid_t'}}

void foo() {
  (void)bar;
}

void test2();
void test2(undef); // expected-error {{a parameter list without types is only allowed in a function definition}}
void test2() { }

void test3();
void test3; // expected-error {{incomplete type}}
void test3() { }

void ellipsis1(...); // expected-error {{ISO C requires a named parameter before '...'}}
