// RUN: clang -fsyntax-only %s -verify

typedef void (^CL)(void);

CL foo() {

	short y;

  short (^add1)(void) = ^{ return y+1; }; // expected-error {{incompatible block pointer types initializing 'int (^)(void)', expected 'short (^)(void)'}}

	CL X = ^{ 
    if (2) 
      return; 
    return 1;  // expected-error {{void block should not return a value}}
  };
	int (^Y) (void)  = ^{ 
    if (3)
      return 1;
    else
      return; // expected-error {{non-void block should return a value}}
  };

	char *(^Z)(void) = ^{ 
    if (3)
      return "";
    else
      return (char*)0;
  };

  double (^A)(void) = ^ { // expected-error {{incompatible block pointer types initializing 'float (^)(void)', expected 'double (^)(void)'}}
    if (1)	
      return (float)1.0; 
    else
      if (2)
       return (double)2.0; // expected-error {{incompatible type returning 'double', expected 'float'}}
    return 1; // expected-error {{incompatible type returning 'int', expected 'float'}}
  };
  
  char *(^B)(void) = ^{ 
    if (3)
      return "";
    else
      return 2; // expected-error {{incompatible type returning 'int', expected 'char *'}}
  };
  return ^{ return 1; }; // expected-error {{incompatible block pointer types returning 'int (^)(void)', expected 'CL'}} expected-error {{returning block that lives on the local stack}}
}

typedef int (^CL2)(void);

CL2 foo2() {
  return ^{ return 1; }; // expected-error {{returning block that lives on the local stack}}
}
