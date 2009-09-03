// RUN: clang-cc -fsyntax-only -verify %s 

struct A {
   A() : value(), cvalue() { } // expected-error {{cannot initialize the member to null in default constructor because reference member 'value' cannot be null-initialized}} \
                               // expected-error {{constructor for 'struct A' must explicitly initialize the reference member 'value'}}
   int &value;	// expected-note{{declared at}}	 {{expected-note{{declared at}}
   const int cvalue;
};

struct B {
};

struct X {
   X() { }	// expected-error {{constructor for 'struct X' must explicitly initialize the reference member 'value'}} \
		// expected-error {{constructor for 'struct X' must explicitly initialize the const member 'cvalue'}} \
		// expected-error {{constructor for 'struct X' must explicitly initialize the reference member 'b'}} \
		// expected-error {{constructor for 'struct X' must explicitly initialize the const member 'cb'}}
   int &value; // expected-note{{declared at}}
   const int cvalue; // expected-note{{declared at}}
   B& b; // expected-note{{declared at}}
   const B cb; // expected-note{{declared at}}
};
