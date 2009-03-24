// RUN: clang-cc -rewrite-objc %s -o=-

@protocol P @end

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

int LOOP();
int INNERLOOP();
void END_LOOP();
@implementation MyList (BasicTest)
- (void)compilerTestAgainst {
  id el;
        for (el in self) 
	  { LOOP(); 
            for (id el1 in self) 
	       INNER_LOOP();

	    END_LOOP();
	  }
}
@end

