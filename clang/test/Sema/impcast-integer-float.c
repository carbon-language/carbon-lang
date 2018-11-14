// RUN: %clang_cc1 %s -verify -Wfloat-precision -fsyntax-only

#define shift_plus_one(x) ((1ULL << x) + 1)

void test(void) {
    float a1 = (1ULL << 31) + 1; // expected-warning {{implicit conversion from 'unsigned long long' to 'float' changes value from 2147483649 to 2.1474836E+9}}
    float a2 = 1ULL << 31;
    float a3 = shift_plus_one(31); // expected-warning {{implicit conversion from 'unsigned long long' to 'float' changes value from 2147483649 to 2.1474836E+9}}
    float a4 = (1ULL << 31) - 1; // expected-warning {{implicit conversion from 'unsigned long long' to 'float' changes value from 2147483647 to 2.1474836E+9}}

    double b1 = (1ULL << 63) + 1; // expected-warning {{implicit conversion from 'unsigned long long' to 'double' changes value from 9223372036854775809 to 9.223372036854775E+18}}
    double b2 = 1ULL << 63;
    double b3 = shift_plus_one(63); // expected-warning {{implicit conversion from 'unsigned long long' to 'double' changes value from 9223372036854775809 to 9.223372036854775E+18}}
    double b4 = (1ULL << 63) - 1; // expected-warning {{implicit conversion from 'unsigned long long' to 'double' changes value from 9223372036854775807 to 9.223372036854775E+18}} 
    
    long double c1 = ((__int128)1 << 127) + 1; // expected-warning {{implicit conversion from '__int128' to 'long double' changes value from -170141183460469231731687303715884105727 to -1.7014118346046923173E+38}}
    long double c2 = (__int128)1 << 127;

    _Float16 d1 = (1ULL << 15) + 1; // expected-warning {{implicit conversion from 'unsigned long long' to '_Float16' changes value from 32769 to 3.277E+4}}
    _Float16 d2 = 1ULL << 15;
    _Float16 d3 = shift_plus_one(15); // expected-warning {{implicit conversion from 'unsigned long long' to '_Float16' changes value from 32769 to 3.277E+4}}
    _Float16 d4 = (1ULL << 15) - 1; // expected-warning {{implicit conversion from 'unsigned long long' to '_Float16' changes value from 32767 to 3.277E+4}}

    float e = (__uint128_t)-1; // expected-warning {{implicit conversion from '__uint128_t' (aka 'unsigned __int128') to 'float' changes value from 340282366920938463463374607431768211455 to +Inf}}
}
// RUN: %clang_cc1 %s -verify -Wfloat-precision -fsyntax-only

#define shift_plus_one(x) ((1ULL << x) + 1)

void test(void) {
    float a1 = (1ULL << 31) + 1; // expected-warning {{implicit conversion from 'unsigned long long' to 'float' changes value from 2147483649 to 2.1474836E+9}}
    float a2 = 1ULL << 31;
    float a3 = shift_plus_one(31); // expected-warning {{implicit conversion from 'unsigned long long' to 'float' changes value from 2147483649 to 2.1474836E+9}}
    float a4 = (1ULL << 31) - 1; // expected-warning {{implicit conversion from 'unsigned long long' to 'float' changes value from 2147483647 to 2.1474836E+9}}

    double b1 = (1ULL << 63) + 1; // expected-warning {{implicit conversion from 'unsigned long long' to 'double' changes value from 9223372036854775809 to 9.223372036854775E+18}}
    double b2 = 1ULL << 63;
    double b3 = shift_plus_one(63); // expected-warning {{implicit conversion from 'unsigned long long' to 'double' changes value from 9223372036854775809 to 9.223372036854775E+18}}
    double b4 = (1ULL << 63) - 1; // expected-warning {{implicit conversion from 'unsigned long long' to 'double' changes value from 9223372036854775807 to 9.223372036854775E+18}} 
    
    long double c1 = ((__int128)1 << 127) + 1; // expected-warning {{implicit conversion from '__int128' to 'long double' changes value from -170141183460469231731687303715884105727 to -1.7014118346046923173E+38}}
    long double c2 = (__int128)1 << 127;

    _Float16 d1 = (1ULL << 15) + 1; // expected-warning {{implicit conversion from 'unsigned long long' to '_Float16' changes value from 32769 to 3.277E+4}}
    _Float16 d2 = 1ULL << 15;
    _Float16 d3 = shift_plus_one(15); // expected-warning {{implicit conversion from 'unsigned long long' to '_Float16' changes value from 32769 to 3.277E+4}}
    _Float16 d4 = (1ULL << 15) - 1; // expected-warning {{implicit conversion from 'unsigned long long' to '_Float16' changes value from 32767 to 3.277E+4}}

    float e = (__uint128_t)-1; // expected-warning {{implicit conversion from '__uint128_t' (aka 'unsigned __int128') to 'float' changes value from 340282366920938463463374607431768211455 to +Inf}}
}
