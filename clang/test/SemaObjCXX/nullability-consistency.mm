// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs %s -verify

#include "nullability-consistency-1.h"
#include "nullability-consistency-3.h"
#include "nullability-consistency-4.h"
#include "nullability-consistency-5.h"
#include "nullability-consistency-5.h"
#include "nullability-consistency-6.h"
#include "nullability-consistency-7.h"
#include "nullability-consistency-8.h"

void h1(int *ptr) { } // don't warn

void h2(int * __nonnull) { }
