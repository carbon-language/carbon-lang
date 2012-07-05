#!/bin/bash
set -e

SRCS="
	tsan_go.cc
	../rtl/tsan_clock.cc
	../rtl/tsan_flags.cc
	../rtl/tsan_md5.cc
	../rtl/tsan_mutex.cc
	../rtl/tsan_platform_linux.cc
	../rtl/tsan_printf.cc
	../rtl/tsan_report.cc
	../rtl/tsan_rtl.cc
	../rtl/tsan_rtl_mutex.cc
	../rtl/tsan_rtl_report.cc
	../rtl/tsan_rtl_thread.cc
	../rtl/tsan_stat.cc
	../rtl/tsan_suppressions.cc
	../rtl/tsan_symbolize.cc
	../rtl/tsan_sync.cc
	../../sanitizer_common/sanitizer_allocator.cc
	../../sanitizer_common/sanitizer_common.cc
	../../sanitizer_common/sanitizer_libc.cc
	../../sanitizer_common/sanitizer_linux.cc
	../../sanitizer_common/sanitizer_posix.cc
	../../sanitizer_common/sanitizer_printf.cc
	../../sanitizer_common/sanitizer_symbolizer.cc
"

#ASMS="../rtl/tsan_rtl_amd64.S"

rm -f gotsan.cc
for F in $SRCS; do
	cat $F >> gotsan.cc
done

CFLAGS=" -I../rtl -I../.. -I../../sanitizer_common -fPIC -g -Wall -Werror -ffreestanding -fno-exceptions -DTSAN_GO -DSANITIZER_GO"
if [ "$DEBUG" == "" ]; then
	CFLAGS+=" -DTSAN_DEBUG=0 -O3 -fno-omit-frame-pointer"
else
	CFLAGS+=" -DTSAN_DEBUG=1 -g"
fi

echo gcc gotsan.cc -S -o tmp.s $CFLAGS
gcc gotsan.cc -S -o tmp.s $CFLAGS
cat tmp.s $ASMS > gotsan.s
echo as gotsan.s -o gotsan.syso
as gotsan.s -o gotsan.syso

