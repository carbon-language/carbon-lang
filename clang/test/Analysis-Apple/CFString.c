// RUN: clang -checker-cfref -verify %s

#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFArray.h>

void f1() {
  
  // Create the array.
  CFMutableArrayRef A = CFArrayCreateMutable(NULL, 10, &kCFTypeArrayCallBacks);

  // Create a string.
  CFStringRef s1 = CFStringCreateWithCString(NULL, "hello world",
                                             kCFStringEncodingUTF8);

  // Add the string to the array.
  CFArrayAppendValue(A, s1);
  
  // Decrement the reference count.
  CFRelease(s1); // no-warning
  
  // Get the string.  We don't own it.
  s1 = (CFStringRef) CFArrayGetValueAtIndex(A, 0);
  
  // Release the array.
  CFRelease(A); // no-warning
  
  // Release the string.  This is a bug.
  CFRelease(s1); // expected-warning{{Incorrect decrement of the reference count}}
}

