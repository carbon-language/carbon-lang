//===-- stacker_rt.c - Runtime Support For Stacker Compiler -----*- C++ -*-===//
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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>

extern int64_t _index_;
extern int64_t _stack_[];
extern void _MAIN_();

void
_stacker_dump_stack_()
{
    int64_t i;
    printf("Stack Dump:\n");
    for (i = _index_; i > 0; i-- )
    {
	printf("#%03lld: %lld\n", (long long int) i, (long long int) _stack_[i] );
    }
}

int
main ( int argc, char** argv )
{
    // Avoid modifying argc
    int a = argc;

    // Make sure we're starting with the right index
    _index_ = 0;

    // Copy the arguments to the stack in reverse order
    // so that they get popped in the order presented
    while ( a > 0 )
    {
	if ( isdigit( (int) argv[--a][0] ) )
	{
	    _stack_[_index_++] = atoll( argv[a] );
	}
	else
	{
	    _stack_[_index_++] = (int64_t) (intptr_t) argv[a];
	}
    }

    // Put the argument count on the stack
    _stack_[_index_] = argc;

    // Invoke the user's main program
    _MAIN_();

    // Return last item on the stack
    if ( _index_ >= 0 )
	return _stack_[_index_];
    return -1;
}
