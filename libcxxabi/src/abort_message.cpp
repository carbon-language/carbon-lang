//===-------------------------- abort_message.c ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#if __APPLE__ 
    #include <Availability.h>
    #if __MAC_OS_X_VERSION_MIN_REQUIRED >= 1070
        #include <CrashReporterClient.h>

        //  If any clients of llvm try to link to libCrashReporterClient.a themselves,
        //  only one crash info struct will be used.
        extern "C" {
        CRASH_REPORTER_CLIENT_HIDDEN 
        struct crashreporter_annotations_t gCRAnnotations 
                __attribute__((section("__DATA," CRASHREPORTER_ANNOTATIONS_SECTION))) 
                = { CRASHREPORTER_ANNOTATIONS_VERSION, 0, 0, 0, 0, 0, 0 };
        }

    #endif
#endif

#include "abort_message.h"

__attribute__((visibility("hidden")))
void abort_message(const char* format, ...)
{
	// write message to stderr
#if __APPLE__
	fprintf(stderr, "libc++abi.dylib: ");
#endif
	va_list	list;
	va_start(list, format);
	vfprintf(stderr, format, list);
	va_end(list);
	fprintf(stderr, "\n");
	
#if __APPLE__ && (__MAC_OS_X_VERSION_MIN_REQUIRED >= 1070)
	// record message in crash report
	char* buffer;
	va_list	list2;
	va_start(list2, format);
	vasprintf(&buffer, format, list2);
	va_end(list2);
	CRSetCrashLogMessage(buffer);
#endif

	abort();
}

