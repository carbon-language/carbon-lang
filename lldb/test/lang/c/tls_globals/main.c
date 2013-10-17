//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void shared_check();

// Create some TLS storage within the static executable.
__thread int var_static = 44;

void *fn_static(void *param)
{
	var_static *= 2;
	shared_check();
	usleep(1); // thread breakpoint
	for(;;)
		usleep(1);
}

int main (int argc, char const *argv[])
{
	pthread_t handle;
	pthread_create(&handle, NULL, &fn_static, NULL);

	for(;;)
		usleep(1); // main breakpoint

	return 0;
}
