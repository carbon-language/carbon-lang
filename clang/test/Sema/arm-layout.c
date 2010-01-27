// RUN: %clang_cc1 -triple armv7-unknown-unknown -target-abi apcs-gnu %s -verify
// RUN: %clang_cc1 -triple armv7-unknown-unknown -target-abi aapcs %s -verify

#ifdef __ARM_EABI__

struct s0 { char field0; double field1; };
int g0[sizeof(struct s0) == 16 ? 1 : -1];

struct s1 { char field0; long double field1; };
int g1[sizeof(struct s1) == 16 ? 1 : -1];

#else

struct s0 { char field0; double field1; };
int g0[sizeof(struct s0) == 12 ? 1 : -1];

struct s1 { char field0; long double field1; };
int g1[sizeof(struct s1) == 12 ? 1 : -1];

#endif
