//===-- stacker_rt.c - Runtime Suppor For Stacker Compiler ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and donated to the LLVM research 
// group and is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This file defines a stack dumping function that can be used for debugging.
//  It is called whenever the DUMP built-in word is used in the Stacker source.
//  It has no effect on the stack (other than to print it).
//
//  The real reason this is here is to test LLVM's ability to link with
//  separately compiled software.
//
//===----------------------------------------------------------------------===//

#include "stdio.h"

extern long _index_;
extern int _stack_[1024];

void
_stacker_dump_stack_()
{
    int i;
    printf("Stack Dump:\n");
    for (i = _index_; i > 0; i-- )
    {
	printf("#%03d: %d\n", i, _stack_[i] );
    }
}
