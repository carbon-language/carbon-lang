// Check the various ways in which the three classes of values
// (scalar, complex, aggregate) interact with parameter passing
// (function entry, function return, call argument, call result).

// RUN: clang %s -triple i386-unknown-unknown -O3 -emit-llvm -o %t &&
// RUN: not grep '@g0' %t &&

// FIXME: Enable once PR3489 is fixed.
// RUNX: clang %s -triple x86_64-unknown-unknown -O3 -emit-llvm -o %t &&
// RUNX: not grep '@g0' %t &&

// RUN: clang %s -triple ppc-unknown-unknown -O3 -emit-llvm -o %t &&
// RUN: not grep '@g0' %t &&
// RUN: true

typedef int ScalarTy;
typedef _Complex int ComplexTy;
typedef struct { int a, b, c; } AggrTy;

static int result;

static AggrTy aggr_id(AggrTy a) { return a; }
static ScalarTy scalar_id(ScalarTy a) { return a; }
static ComplexTy complex_id(ComplexTy a) { return a; }

static void aggr_mul(AggrTy a) { result *= a.a * a.b * a.c; }

static void scalar_mul(ScalarTy a) { result *= a; }

static void complex_mul(ComplexTy a) { result *= __real a * __imag a; }

extern void g0(void);

void f0(void) {
  result = 1;
  
  aggr_mul(aggr_id((AggrTy) { 2, 3, 5}));
  scalar_mul(scalar_id(7));
  complex_mul(complex_id(11 + 13i));
  
  // This call should be eliminated.
  if (result != 30030)
    g0();
}

