// RUN: clang -fsyntax-only -verify %s

void * proc();

@interface NSConstantString
@end

@interface Frob
@end

@interface Frob1
@end

void * foo()
{
  @try {
    return proc();
  }
  @catch (Frob* ex) {
    @throw;
  }
  @catch (Frob1* ex) {
    @throw proc();
  }
  @finally {
    @try {
      return proc();
    }
    @catch (Frob* ex) {
      @throw 1,2;
    }
    @catch(...) {
      @throw (4,3,proc());
    }
  }

  @try {  // expected-error {{@try statement without a @catch and @finally clause}}
    return proc();
  }
}


void bar()
{
  @try {}// expected-error {{@try statement without a @catch and @finally clause}}
  @"s"; //  expected-warning {{result unused}}
}

void baz()
{
  @try {}// expected-error {{@try statement without a @catch and @finally clause}}
  @try {}
  @finally {}
}

void noTwoTokenLookAheadRequiresABitOfFancyFootworkInTheParser() {
    @try {
        // Do something
    } @catch (...) {}
    @try {
        // Do something
    } @catch (...) {}
    return 0;
}

