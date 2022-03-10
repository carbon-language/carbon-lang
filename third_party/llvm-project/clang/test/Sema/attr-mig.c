// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int kern_return_t;
#define KERN_SUCCESS 0

__attribute__((mig_server_routine)) kern_return_t var = KERN_SUCCESS; // expected-warning{{'mig_server_routine' attribute only applies to functions, Objective-C methods, and blocks}}

__attribute__((mig_server_routine)) void foo_void(void); // expected-warning{{'mig_server_routine' attribute only applies to routines that return a kern_return_t}}
__attribute__((mig_server_routine)) int foo_int(void); // expected-warning{{'mig_server_routine' attribute only applies to routines that return a kern_return_t}}

__attribute__((mig_server_routine)) kern_return_t bar_extern(void); // no-warning
__attribute__((mig_server_routine)) kern_return_t bar_forward(void); // no-warning

__attribute__((mig_server_routine)) kern_return_t bar_definition(void) { // no-warning
  return KERN_SUCCESS;
}

kern_return_t bar_forward(void) { // no-warning
  return KERN_SUCCESS;
}

__attribute__((mig_server_routine(123))) kern_return_t bar_with_argument(void); // expected-error{{'mig_server_routine' attribute takes no arguments}}
