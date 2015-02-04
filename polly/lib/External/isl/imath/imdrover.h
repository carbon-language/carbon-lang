/*
  Name:     imdrover.h
  Purpose:  Keeper of the hordes of testing code.
  Author:   M. J. Fromberger <http://spinning-yarns.org/michael/>

  Copyright (C) 2002-2007 Michael J. Fromberger, All Rights Reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
 */

#ifndef IMDROVER_H_
#define IMDROVER_H_

#include <stdio.h>

typedef struct {
  int    line;
  char  *code;
  int    num_inputs;
  char **input;
  int    num_outputs;
  char **output;
} testspec_t;

typedef int (*test_f)(testspec_t *, FILE *);

/* Call this once at the outset to set up test registers */
void init_testing(void);
void reset_registers(void);

/* Integer tests, and general */
int test_init(testspec_t* t, FILE* ofp);
int test_set(testspec_t* t, FILE* ofp);
int test_neg(testspec_t* t, FILE* ofp);
int test_abs(testspec_t* t, FILE* ofp);
int test_add(testspec_t* t, FILE* ofp);
int test_sub(testspec_t* t, FILE* ofp);
int test_mul(testspec_t* t, FILE* ofp);
int test_mulp2(testspec_t* t, FILE* ofp);
int test_mulv(testspec_t* t, FILE* ofp);
int test_sqr(testspec_t* t, FILE* ofp);
int test_div(testspec_t* t, FILE* ofp);
int test_divp2(testspec_t* t, FILE* ofp);
int test_divv(testspec_t* t, FILE* ofp);
int test_expt(testspec_t* t, FILE* ofp);
int test_exptv(testspec_t* t, FILE* ofp);
int test_exptf(testspec_t* t, FILE* ofp);
int test_mod(testspec_t* t, FILE* ofp);
int test_gcd(testspec_t* t, FILE* ofp);
int test_egcd(testspec_t* t, FILE* ofp);
int test_lcm(testspec_t* t, FILE* ofp);
int test_sqrt(testspec_t* t, FILE* ofp);
int test_root(testspec_t* t, FILE* ofp);
int test_invmod(testspec_t* t, FILE* ofp);
int test_exptmod(testspec_t* t, FILE* ofp);
int test_exptmod_ev(testspec_t* t, FILE* ofp);
int test_exptmod_bv(testspec_t* t, FILE* ofp);
int test_comp(testspec_t* t, FILE* ofp);
int test_ucomp(testspec_t* t, FILE* ofp);
int test_zcomp(testspec_t* t, FILE* ofp);
int test_vcomp(testspec_t* t, FILE* ofp);
int test_uvcomp(testspec_t* t, FILE* ofp);
int test_tostr(testspec_t* t, FILE* ofp);
int test_tobin(testspec_t* t, FILE* ofp);
int test_to_int(testspec_t* t, FILE* ofp);
int test_to_uint(testspec_t* t, FILE* ofp);
int test_read_binary(testspec_t* t, FILE* ofp);
int test_to_uns(testspec_t* t, FILE* ofp);
int test_read_uns(testspec_t* t, FILE* ofp);
int test_meta(testspec_t* t, FILE* ofp);

/* Rational tests */
int test_qneg(testspec_t* t, FILE* ofp);
int test_qrecip(testspec_t* t, FILE* ofp);
int test_qabs(testspec_t* t, FILE* ofp);
int test_qadd(testspec_t* t, FILE* ofp);
int test_qsub(testspec_t* t, FILE* ofp);
int test_qmul(testspec_t* t, FILE* ofp);
int test_qdiv(testspec_t* t, FILE* ofp);
int test_qdiv(testspec_t* t, FILE* ofp);
int test_qaddz(testspec_t* t, FILE* ofp);
int test_qsubz(testspec_t* t, FILE* ofp);
int test_qmulz(testspec_t* t, FILE* ofp);
int test_qdivz(testspec_t* t, FILE* ofp);
int test_qexpt(testspec_t* t, FILE* ofp);
int test_qtostr(testspec_t* t, FILE* ofp);
int test_qtodec(testspec_t* t, FILE* ofp);
int test_qrdec(testspec_t* t, FILE* ofp);

#endif /* IMDROVER_H_ */
