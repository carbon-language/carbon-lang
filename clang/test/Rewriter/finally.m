// RUN: clang -rewrite-objc -verify %s -o -

int main() {
  @try {
    printf("executing try");
    return(0); // expected-warning{{rewriter doesn't support user-specified control flow semantics for @try/@finally (code may not execute properly)}}
  } @finally {
    printf("executing finally");
  }
  while (1) {
    @try {
      printf("executing try");
      break; // expected-warning{{rewriter doesn't support user-specified control flow semantics for @try/@finally (code may not execute properly)}}
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

