// RUN: clang -fsyntax-only -verify %s

void * proc();

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

	@try {  // expected-error {{@try statment without a @catch and @finally clause}}
                return proc();
        }
}

