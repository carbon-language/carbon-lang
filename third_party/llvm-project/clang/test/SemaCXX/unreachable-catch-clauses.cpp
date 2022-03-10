// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

class BaseEx {};
class Ex1: public BaseEx {};
typedef Ex1 Ex2;

void f();

void test()
try {}
catch (BaseEx &e) { f(); } // expected-note 2{{for type 'BaseEx &'}}
catch (Ex1 &e) { f(); } // expected-warning {{exception of type 'Ex1 &' will be caught by earlier handler}}
catch (Ex2 &e) { f(); } // expected-warning {{exception of type 'Ex2 &' (aka 'Ex1 &') will be caught by earlier handler}}
