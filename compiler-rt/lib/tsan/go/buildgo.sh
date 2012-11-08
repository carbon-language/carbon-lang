#!/bin/bash
set -e
set -x

SRCS="
	tsan_go.cc
	../rtl/tsan_clock.cc
	../rtl/tsan_flags.cc
	../rtl/tsan_md5.cc
	../rtl/tsan_mutex.cc
	../rtl/tsan_report.cc
	../rtl/tsan_rtl.cc
	../rtl/tsan_rtl_mutex.cc
	../rtl/tsan_rtl_report.cc
	../rtl/tsan_rtl_thread.cc
	../rtl/tsan_stat.cc
	../rtl/tsan_suppressions.cc
	../rtl/tsan_sync.cc
	../../sanitizer_common/sanitizer_allocator.cc
	../../sanitizer_common/sanitizer_common.cc
	../../sanitizer_common/sanitizer_flags.cc
	../../sanitizer_common/sanitizer_libc.cc
	../../sanitizer_common/sanitizer_printf.cc
"

if [ "`uname -a | grep Linux`" != "" ]; then
	SUFFIX="linux_amd64"
	OSCFLAGS="-fPIC -ffreestanding"
	OSLDFLAGS="-lpthread -fPIC -fpie"
	SRCS+="
		../rtl/tsan_platform_linux.cc
		../../sanitizer_common/sanitizer_posix.cc
		../../sanitizer_common/sanitizer_linux.cc
	"
elif [ "`uname -a | grep Darwin`" != "" ]; then
	SUFFIX="darwin_amd64"
	OSCFLAGS="-fPIC"
	OSLDFLAGS="-lpthread -fPIC -fpie"
	SRCS+="
		../rtl/tsan_platform_mac.cc
		../../sanitizer_common/sanitizer_posix.cc
		../../sanitizer_common/sanitizer_mac.cc
	"
elif [ "`uname -a | grep MINGW`" != "" ]; then
	SUFFIX="windows_amd64"
	OSCFLAGS="-Wno-error=attributes -Wno-attributes"
	OSLDFLAGS=""
	SRCS+="
		../rtl/tsan_platform_windows.cc
		../../sanitizer_common/sanitizer_win.cc
	"
else
	echo Unknown platform
	exit 1
fi

SRCS+=$ADD_SRCS

rm -f gotsan.cc
for F in $SRCS; do
	cat $F >> gotsan.cc
done

FLAGS=" -I../rtl -I../.. -I../../sanitizer_common -I../../../include -m64 -Wall -Werror -fno-exceptions -DTSAN_GO -DSANITIZER_GO -DTSAN_SHADOW_COUNT=4 $OSCFLAGS"
if [ "$DEBUG" == "" ]; then
	FLAGS+=" -DTSAN_DEBUG=0 -O3 -fomit-frame-pointer"
else
	FLAGS+=" -DTSAN_DEBUG=1 -g"
fi

echo gcc gotsan.cc -S -o tmp.s $FLAGS $CFLAGS
gcc gotsan.cc -S -o tmp.s $FLAGS $CFLAGS
cat tmp.s $ASMS > gotsan.s
echo as gotsan.s -o race_$SUFFIX.syso
as gotsan.s -o race_$SUFFIX.syso

gcc test.c race_$SUFFIX.syso -m64 -o test $OSLDFLAGS
GORACE="exitcode=0 atexit_sleep_ms=0" ./test
