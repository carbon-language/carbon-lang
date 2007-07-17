// RUN: clang -parse-ast-check %s

typedef float float4 __attribute__((vector_size(16)));
typedef int int4 __attribute__((vector_size(16)));
typedef int4* int4p;

void test1(float4 a, int4 *result, int i) {
    result[i] = a; // expected-error {{assigning 'float4' to 'int4'}}
}

void test2(float4 a, int4p result, int i) {
    result[i] = a; // expected-error {{assigning 'float4' to 'int4'}}
}

