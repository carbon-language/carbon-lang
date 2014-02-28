#!/bin/bash
set -e

SRCS="
	dd_rtl.cc
	dd_interceptors.cc
	../../sanitizer_common/sanitizer_allocator.cc
	../../sanitizer_common/sanitizer_common.cc
	../../sanitizer_common/sanitizer_deadlock_detector1.cc
	../../sanitizer_common/sanitizer_flags.cc
	../../sanitizer_common/sanitizer_libc.cc
	../../sanitizer_common/sanitizer_printf.cc
	../../sanitizer_common/sanitizer_suppressions.cc
	../../sanitizer_common/sanitizer_thread_registry.cc
	../../sanitizer_common/sanitizer_posix.cc
	../../sanitizer_common/sanitizer_posix_libcdep.cc
	../../sanitizer_common/sanitizer_procmaps_linux.cc
	../../sanitizer_common/sanitizer_linux.cc
	../../sanitizer_common/sanitizer_linux_libcdep.cc
	../../sanitizer_common/sanitizer_stoptheworld_linux_libcdep.cc
	../../sanitizer_common/sanitizer_stackdepot.cc
	../../sanitizer_common/sanitizer_stacktrace.cc
	../../sanitizer_common/sanitizer_stacktrace_libcdep.cc
	../../sanitizer_common/sanitizer_symbolizer.cc
	../../sanitizer_common/sanitizer_symbolizer_libcdep.cc
	../../sanitizer_common/sanitizer_symbolizer_posix_libcdep.cc
	../../sanitizer_common/sanitizer_symbolizer_libbacktrace.cc
	../../interception/interception_linux.cc
"

FLAGS=" -I../.. -I../../sanitizer_common -I../../interception -Wall -fno-exceptions -fno-rtti -DSANITIZER_USE_MALLOC"
if [ "$DEBUG" == "" ]; then
	FLAGS+=" -DDEBUG=0 -O3 -fomit-frame-pointer"
else
	FLAGS+=" -DDEBUG=1 -g"
fi

rm -f dd.cc
for F in $SRCS; do
	g++ $F -c -o dd.o $FLAGS
	cat $F >> dd.cc
done

g++ dd.cc -c -o dd.o $FLAGS

