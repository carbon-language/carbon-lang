// RUN: %clang_cc1 -rewrite-objc -verify %s -o -

int main() {
  @try {
    printf("executing try"); // expected-warning{{implicitly declaring C library function 'printf' with type 'int (char const *, ...)'}} \
        // expected-note{{please include the header <stdio.h> or explicitly provide a declaration for 'printf'}}
    return(0); // expected-warning{{rewriter doesn't support user-specified control flow semantics for @try/@finally (code may not execute properly)}}
  } @finally {
    printf("executing finally");
  }
  while (1) {
    @try {
      printf("executing try");
      break;
    } @finally {
      printf("executing finally");
    }
    printf("executing after finally block");
  }
  @try {
    printf("executing try");
  } @finally {
    printf("executing finally");
  }
  return 0;
}

void test_sync_with_implicit_finally() {
    id foo;
    @synchronized (foo) {
        return; // The rewriter knows how to generate code for implicit finally
    }
}

void test2_try_with_implicit_finally() {
    @try {
        return; // The rewriter knows how to generate code for implicit finally
    } @catch (id e) {
        
    }
}

