/*
 * =============================================================================
 *
 *       Filename:  20120330-argument-use.c
 *
 *    Description:  Polly OpenMP test case
 *
 *                  Test if the OpenMP subfunction uses the argument copy in
 *                  the OpenMP struct not the original one only available in
 *                  the original function.
 *
 *                  Run with -polly-codegen -enable-polly-openmp
 *
 *         Author:  Johannes Doerfert johannes@jdoerfert.de
 *
 *        Created:  2012-03-30
 *       Modified:  2012-03-30
 *
 * =============================================================================
 */

void f(int * restrict A, int * restrict B, int n) {
  int i;

  for (i = 0; i < n; i++) {
    A[i] = B[i] * 2;
  }
}
