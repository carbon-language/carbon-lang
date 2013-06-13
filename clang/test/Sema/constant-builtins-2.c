// RUN: %clang_cc1 -fsyntax-only -verify %s

// Math stuff

double       g0  = __builtin_huge_val();
float        g1  = __builtin_huge_valf();
long double  g2  = __builtin_huge_vall();

double       g3  = __builtin_inf();
float        g4  = __builtin_inff();
long double  g5  = __builtin_infl();

double       g6  = __builtin_nan("");
float        g7  = __builtin_nanf("");
long double  g8  = __builtin_nanl("");

// GCC constant folds these too (via native strtol):
//double       g6_1  = __builtin_nan("1");
//float        g7_1  = __builtin_nanf("1");
//long double  g8_1  = __builtin_nanl("1");

// APFloat doesn't have signalling NaN functions.
//double       g9  = __builtin_nans("");
//float        g10 = __builtin_nansf("");
//long double  g11 = __builtin_nansl("");

//int          g12 = __builtin_abs(-12);

double       g13 = __builtin_fabs(-12.);
double       g13_0 = __builtin_fabs(-0.);
double       g13_1 = __builtin_fabs(-__builtin_inf());
float        g14 = __builtin_fabsf(-12.f);
// GCC doesn't eat this one.
//long double  g15 = __builtin_fabsfl(-12.0L);

float        g16 = __builtin_copysign(1.0, -1.0);
double       g17 = __builtin_copysignf(1.0f, -1.0f);
long double  g18 = __builtin_copysignl(1.0L, -1.0L);

//double       g19 = __builtin_powi(2.0, 4);
//float        g20 = __builtin_powif(2.0f, 4);
//long double  g21 = __builtin_powil(2.0L, 4);

#define BITSIZE(x) (sizeof(x) * 8)
char g22[__builtin_clz(1) == BITSIZE(int) - 1 ? 1 : -1];
char g23[__builtin_clz(7) == BITSIZE(int) - 3 ? 1 : -1];
char g24[__builtin_clz(1 << (BITSIZE(int) - 1)) == 0 ? 1 : -1];
int g25 = __builtin_clz(0); // expected-error {{not a compile-time constant}}
char g26[__builtin_clzl(0xFL) == BITSIZE(long) - 4 ? 1 : -1];
char g27[__builtin_clzll(0xFFLL) == BITSIZE(long long) - 8 ? 1 : -1];

char g28[__builtin_ctz(1) == 0 ? 1 : -1];
char g29[__builtin_ctz(8) == 3 ? 1 : -1];
char g30[__builtin_ctz(1 << (BITSIZE(int) - 1)) == BITSIZE(int) - 1 ? 1 : -1];
int g31 = __builtin_ctz(0); // expected-error {{not a compile-time constant}}
char g32[__builtin_ctzl(0x10L) == 4 ? 1 : -1];
char g33[__builtin_ctzll(0x100LL) == 8 ? 1 : -1];

char g34[__builtin_popcount(0) == 0 ? 1 : -1];
char g35[__builtin_popcount(0xF0F0) == 8 ? 1 : -1];
char g36[__builtin_popcount(~0) == BITSIZE(int) ? 1 : -1];
char g37[__builtin_popcount(~0L) == BITSIZE(int) ? 1 : -1];
char g38[__builtin_popcountl(0L) == 0 ? 1 : -1];
char g39[__builtin_popcountl(0xF0F0L) == 8 ? 1 : -1];
char g40[__builtin_popcountl(~0L) == BITSIZE(long) ? 1 : -1];
char g41[__builtin_popcountll(0LL) == 0 ? 1 : -1];
char g42[__builtin_popcountll(0xF0F0LL) == 8 ? 1 : -1];
char g43[__builtin_popcountll(~0LL) == BITSIZE(long long) ? 1 : -1];
#undef BITSIZE

// GCC misc stuff

extern int f();

int h0 = __builtin_types_compatible_p(int, float);
//int h1 = __builtin_choose_expr(1, 10, f());
//int h2 = __builtin_expect(0, 0);
int h3 = __builtin_bswap16(0x1234) == 0x3412 ? 1 : f();
int h4 = __builtin_bswap32(0x1234) == 0x34120000 ? 1 : f();
int h5 = __builtin_bswap64(0x1234) == 0x3412000000000000 ? 1 : f();
extern long int bi0;
extern __typeof__(__builtin_expect(0, 0)) bi0;

// Strings
int array1[__builtin_strlen("ab\0cd")];
int array2[(sizeof(array1)/sizeof(int)) == 2? 1 : -1];
