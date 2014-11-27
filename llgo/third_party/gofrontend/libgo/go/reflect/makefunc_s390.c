// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "go-panic.h"

#ifdef __s390x__
#  define S390_GO_USE_64_BIT_ABI 1
#  define S390_GO_S390X_ARGS , double f4, double f6
#  define S390_GO_S390X_FIELDS double f4; double f6;
   extern void S390xMakeFuncStubGo(void *, void *)
	asm ("reflect.S390xMakeFuncStubGo");
#  define S390_GO_MakeFuncStubGo(r, c) S390xMakeFuncStubGo((r), (c))
#else
#  define S390_GO_USE_64_BIT_ABI 0
#  define S390_GO_S390X_ARGS
#  define S390_GO_S390X_FIELDS
   extern void S390MakeFuncStubGo(void *, void *)
	asm ("reflect.S390MakeFuncStubGo");
#  define S390_GO_MakeFuncStubGo(r, c) S390MakeFuncStubGo((r), (c))
   /* Needed to make the unused 64 bit abi conditional code compile.  */
#  define f4 f0
#  define f6 f2
#endif

/* Structure to store all registers used for parameter passing.  */
typedef struct
{
	long r2;
	long r3;
	long r4;
	long r5;
	long r6;
	/* Pointer to non-register arguments on the stack.  */
	long stack_args;
	double f0;
	double f2;
	S390_GO_S390X_FIELDS
} s390Regs;

void
makeFuncStub(long r2, long r3, long r4, long r5, long r6,
	unsigned long stack_args, double f0, double f2
	S390_GO_S390X_ARGS)
	asm ("reflect.makeFuncStub");

void
makeFuncStub(long r2, long r3, long r4, long r5, long r6,
	unsigned long stack_args, double f0, double f2
	S390_GO_S390X_ARGS)
{
	s390Regs regs;
	void *closure;

	/* Store the registers in a structure that is passed on to the Go stub
	   function.  */
	regs.r2 = r2;
	regs.r3 = r3;
	regs.r4 = r4;
	regs.r5 = r5;
	regs.r6 = r6;
	regs.stack_args = (long)&stack_args;
	regs.f0 = f0;
	regs.f2 = f2;
	if (S390_GO_USE_64_BIT_ABI) {
		regs.f4 = f4;
		regs.f6 = f6;
	}
	/* For MakeFunc functions that call recover.  */
	__go_makefunc_can_recover(__builtin_return_address(0));
	/* Call the Go stub function.  */
	closure = __go_get_closure();
	S390_GO_MakeFuncStubGo(&regs, closure);
	/* MakeFunc functions can no longer call recover.  */
	__go_makefunc_returning();
	/* Restore all possible return registers.  */
	if (S390_GO_USE_64_BIT_ABI) {
		asm volatile ("lg\t%%r2,0(%0)" : : "a" (&regs.r2) : "r2" );
		asm volatile ("ld\t%%f0,0(%0)" : : "a" (&regs.f0) : "f0" );
	} else {
		asm volatile ("l\t%%r2,0(%0)" : : "a" (&regs.r2) : "r2" );
		asm volatile ("l\t%%r3,0(%0)" : : "a" (&regs.r3) : "r3" );
		asm volatile ("ld\t%%f0,0(%0)" : : "a" (&regs.f0) : "f0" );
	}
}
