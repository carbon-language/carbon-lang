// RUN: clang-cc -fsyntax-only -verify %s

struct X1 { // has no implicit default constructor
   X1(int);
};

struct X2  : X1 {  // expected-note {{'struct X2' declared here}} \
                  //  expected-note {{'struct X2' declared here}}
   X2(int);
};

struct X3 : public X2 {
};
X3 x3;  // expected-error {{cannot define the implicit default constructor for 'struct X3', because base class 'struct X2' does not have any default constructor}}


struct X4 {
  X2 x2; 	// expected-note {{member is declared here}}
  X2 & rx2; // expected-note {{declared at}}
};

X4 x4; // expected-error {{cannot define the implicit default constructor for 'struct X4', because member's type 'struct X2' does not have any default constructor}} \
       // expected-error {{cannot define the implicit default constructor for 'struct X4', because reference member 'rx2' cannot be default-initialized}}


struct Y1 { // has no implicit default constructor
   Y1(int);
};

struct Y2  : Y1 { 
   Y2(int);
   Y2();
};

struct Y3 : public Y2 {
};
Y3 y3; 

struct Y4 {
  Y2 y2; 
};

Y4 y4;

// More tests


struct Z1 {
  int& z;       // expected-note {{declared at}}
  const int c1; // expected-note {{declared at}}
  volatile int v1;
};

Z1 z1;  // expected-error {{cannot define the implicit default constructor for 'struct Z1', because reference member 'z' cannot be default-initialized}} \
        // expected-error {{cannot define the implicit default constructor for 'struct Z1', because const member 'c1' cannot be default-initialized}}

