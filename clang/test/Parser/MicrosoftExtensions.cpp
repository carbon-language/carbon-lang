// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions

/* Microsoft attribute tests */
[repeatable][source_annotation_attribute( Parameter|ReturnValue )]
struct SA_Post{ SA_Post(); int attr; };

[returnvalue:SA_Post( attr=1)] 
int foo1([SA_Post(attr=1)] void *param);

namespace {
  [returnvalue:SA_Post(attr=1)] 
  int foo2([SA_Post(attr=1)] void *param);
}

class T {
  [returnvalue:SA_Post(attr=1)] 
  int foo3([SA_Post(attr=1)] void *param);
};

extern "C" {
  [returnvalue:SA_Post(attr=1)] 
  int foo5([SA_Post(attr=1)] void *param);
}


