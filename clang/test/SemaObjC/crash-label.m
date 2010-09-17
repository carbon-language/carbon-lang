// RUN: %clang_cc1  -fsyntax-only -verify %s

  - (NSDictionary*) _executeScript:(NSString *)source { // expected-error 2 {{expected a type}} \
							// expected-error {{missing context for method declaration}} 
Exit:  [nilArgs release]; // expected-error {{use of undeclared identifier}}
}
- (NSDictionary *) _setupKernelStandardMode:(NSString *)source { // expected-error 2 {{expected a type}} \
expected-error {{missing context for method declaration}} \
expected-note{{to match this '{'}}
  Exit:   if(_ciKernel && !success ) { // expected-error {{use of undeclared identifier}} // expected-error 2 {{expected}} expected-note{{to match this '{'}} expected-error{{use of undeclared identifier 'success'}}
