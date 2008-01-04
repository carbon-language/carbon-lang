// RUN: clang -fsyntax-only -verify %s

typedef struct objc_class *Class;
typedef struct objc_object {
 Class isa;
} *id;
    
            
@interface MyList
@end
    
@implementation MyList
- (unsigned int)countByEnumeratingWithState:  (struct __objcFastEnumerationState *)state objects:  (id *)items count:(unsigned int)stackcount
{
        return 0;
}
@end

@interface MyList (BasicTest)
- (void)compilerTestAgainst;
@end

@implementation MyList (BasicTest)
- (void)compilerTestAgainst {

        int i=0;
        for (int * elem in elem) // expected-error {{selector element is not of valid object type (its type is 'int *')}} \
				    expected-error {{collection expression is not of valid object type (its type is 'int *')}}
           ++i;
        for (i in elem)  // expected-error {{use of undeclared identifier 'elem'}} \
			    expected-error {{selector element is not of valid object type (its type is 'int')}}
           ++i;
        for (id se in i) // expected-error {{collection expression is not of valid object type (its type is 'int')}} 
           ++i;
}
@end

