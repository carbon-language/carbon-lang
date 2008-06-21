// RUN: clang -fsyntax-only -verify %s

ce MyList // expected-error {{expected '=', ',', ';', 'asm', or '__attribute__' after declarator}}
@end


@implementation MyList
- (unsigned int)countByEnumeratingWithState:  (struct __objcFastEnumerationState *)state objects:  (id *)items count:(unsigned int)stackcount
{
     return 0;
}
@end


int LOOP();

@implementation MyList (BasicTest)  // expected-error {{cannot find interface declaration for 'MyList'}}
- (void)compilerTestAgainst {
MyList * el;  // expected-error {{use of undeclared identifier 'MyList'}}
     for (el in @"foo")    // expected-error {{use of undeclared identifier 'el'}}
	  { LOOP(); }
}
@end

