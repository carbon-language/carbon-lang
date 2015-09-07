// Check that blocks executed on the XPC callback queue are handled correctly by
// ASan.

// RUN: %clangxx_asan -O0 %s -o %t -framework Foundation && not %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#include <sanitizer/asan_interface.h>
#include <xpc/xpc.h>

int main() {
  // Set up a 5-second timeout.
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC),
                 dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                 ^{
                   exit(0);
                 });

  xpc_connection_t _connection = xpc_connection_create_mach_service(
      "com.example.non.existing.xpc", NULL, 0);

  xpc_connection_set_event_handler(_connection, ^(xpc_object_t event) {
    char *mem = (char *)malloc(10);
    NSLog(@"mem = %p", mem);

    // Without the XPC API interceptors, this would cause an assertion failure
    // when describing the current thread (XPC callback thread).
    fprintf(stderr, "%s\n", mem + 10);  // BOOM.
    // CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
    // CHECK: READ of size 1 at
    // CHECK: allocated by thread
    // CHECK: {{    #0 0x.* in .*malloc}}
  });

  xpc_connection_resume(_connection);
  dispatch_main();
  return 0;
}
