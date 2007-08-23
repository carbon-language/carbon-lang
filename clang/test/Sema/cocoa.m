// RUN: clang %s -parse-noop -arch ppc
#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#endif

