//===-- a.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

long double
return_long_double (long double value)
{
  float a=2, b=4,c=8, d=16, e=32, f=64, k=128, l=256, add=0;
__asm__ ( "fld %1 ;"
	"fld %2 ;"
	"fld %3 ;"
	"fld %4 ;"
	"fld %5 ;"
	"fld %6 ;"
	"fld %7 ;"
	"fadd ;" : "=g" (add) : "g" (a), "g" (b), "g" (c), "g" (d), "g" (e), "g" (f), "g" (k), "g" (l) );  // Set break point at this line.
  return value;
}

long double
outer_return_long_double (long double value)
{
  long double val = return_long_double(value);
  val *= 2 ;
  return val;
}

long double
outermost_return_long_double (long double value)
{
  long double val = outer_return_long_double(value);
  val *= 2 ;
  return val;
}
