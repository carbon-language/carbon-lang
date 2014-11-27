// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

/* Dummy function for processors that implement MakeFunc using FFI
   rather than having builtin support.  */

void makeFuncStub (void) __asm__ ("reflect.makeFuncStub");

void makeFuncStub (void)
{
  runtime_throw ("impossible call to makeFuncStub");
}
