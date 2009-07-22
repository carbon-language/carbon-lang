// RUN: clang-cc -rewrite-objc %s -o=-

@interface MyList
- (id) allKeys;
@end
    
@implementation MyList
- (unsigned int)countByEnumeratingWithState:  (struct __objcFastEnumerationState *)state objects:  (id *)items count:(unsigned int)stackcount
{
        return 0;
}
- (id) allKeys { return 0; }
@end

@interface MyList (BasicTest)
- (void)compilerTestAgainst;
@end

int LOOP();
@implementation MyList (BasicTest)
- (void)compilerTestAgainst {
  MyList * el;
  int i;
        for (el in [el allKeys]) { 
		for (i = 0; i < 10; i++)
		  if (i == 5)
		   break;

		if (el == 0)
		 break;
		if (el != self)
		  continue;
		LOOP(); 
	  }

        for (id el1 in[el allKeys]) { 
	    LOOP(); 
	    for (el in self) {
	      if (el)
		continue;
	    }
	    if (el1)
	      break;
	  }
}
@end

