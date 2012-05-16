// RUN: %clang_cc1 -fsyntax-only -Wextra-semi -verify %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -Wextra-semi -fixit %t
// RUN: %clang_cc1 -x c++ -Wextra-semi -Werror %t

class A {
  void A1();
  void A2() { }; // expected-warning{{extra ';' after function definition}}
  ; // expected-warning{{extra ';' inside a class}}
  void A3() { };  ;; // expected-warning{{extra ';' after function definition}}
  ;;;;;;; // expected-warning{{extra ';' inside a class}}
  ; // expected-warning{{extra ';' inside a class}}
  ; ;;		 ;  ;;; // expected-warning{{extra ';' inside a class}}
    ;  ; 	;	;  ;; // expected-warning{{extra ';' inside a class}}
  void A4();
};

union B {
  int a1;
  int a2;; // expected-warning{{extra ';' inside a union}}
};

; // expected-warning{{extra ';' outside of a function}}
; ;;// expected-warning{{extra ';' outside of a function}}

