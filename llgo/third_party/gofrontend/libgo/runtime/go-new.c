/* go-new.c -- the generic go new() function.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "go-alloc.h"
#include "runtime.h"
#include "arch.h"
#include "malloc.h"

void *
__go_new (const struct __go_type_descriptor *td, uintptr_t size)
{
  return runtime_mallocgc (size,
			   (uintptr) td | TypeInfo_SingleObject,
			   0);
}

void *
__go_new_nopointers (const struct __go_type_descriptor *td,  uintptr_t size)
{
  return runtime_mallocgc (size,
			   (uintptr) td | TypeInfo_SingleObject,
			   FlagNoScan);
}
