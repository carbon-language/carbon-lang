// RUN: %clang_cc1 %s -fsyntax-only -verify -Wmissing-noreturn -fobjc-exceptions

int test1() {
  id a;
  @throw a;
}

// PR5286
void test2(int a) {
  while (1) {
    if (a)
      return;
  }
}

// PR5286
void test3(int a) {  // expected-warning {{function could be attribute 'noreturn'}}
  while (1) {
    if (a)
      @throw (id)0;
  }
}

// <rdar://problem/4289832> - This code always returns, we should not
//  issue a noreturn warning.
@class NSException;
@class NSString;
NSString *rdar_4289832() {  // no-warning
    @try
    {
        return @"a";
    }
    @catch(NSException *exception)
    {
        return @"b";
    }
    @finally
    {
    }
}

