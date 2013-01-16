// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -fmodules -I %S/Inputs %s -verify

@import MethodPoolA;


// in other file: // expected-note{{using}}




// in other file: expected-note{{also found}}

void testMethod1(id object) {
  [object method1]; 
}

void testMethod2(id object) {
  [object method2:1];
} 

@import MethodPoolB;

void testMethod1Again(id object) {
  [object method1];
}

void testMethod2Again(id object) {
  [object method2:1]; // expected-warning{{multiple methods named 'method2:' found}}
}

void testMethod3(id object) {
  [object method3]; // expected-warning{{instance method '-method3' not found (return type defaults to 'id')}}
}

@import MethodPoolB.Sub;

void testMethod3Again(id object) {
  char *str = [object method3]; // okay: only found in MethodPoolB.Sub
}

@import MethodPoolA.Sub;

void testMethod3AgainAgain(id object) {
  [object method3]; // expected-warning{{multiple methods named 'method3' found}}
  // expected-note@2{{using}}
  // expected-note@2{{also found}}
}
