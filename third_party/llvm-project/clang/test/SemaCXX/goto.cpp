// RUN: %clang_cc1 -fsyntax-only -verify -Wall -Wno-unused-but-set-variable -fblocks %s

// PR9463
double *end;
void f(bool b1, bool b2) {
  {
    do {
      int end = 0;
      if (b2) {
        do {
          goto end;
        } while (b2);
      }
      end = 1;
    } while (b1);
  }

 end:
  return;
}

namespace N {
  float* end;
  void f(bool b1, bool b2) {
    {
      do {
        int end = 0;
        if (b2) {
          do {
            goto end;
          } while (b2);
        }
        end = 1;
      } while (b1);
    }

  end:
    return;
  }
}

void g() {
  end = 1; // expected-error{{incompatible integer to pointer conversion assigning to 'double *' from 'int'}}
}

void h(int end) {
  {
    goto end; // expected-error{{use of undeclared label 'end'}}
  }
}

void h2(int end) {
  {
    __label__ end;
    goto end;

  end:
    ::end = 0;
  }
 end: // expected-warning{{unused label 'end'}}
  end = 1;
}

class X {
public:
  X();
};

void rdar9135994()
{
X:  
    goto X;
}

namespace PR9495 {
  struct NonPOD { NonPOD(); ~NonPOD(); };  
  
  void f(bool b) {
    NonPOD np;
    if (b) {
      goto undeclared; // expected-error{{use of undeclared label 'undeclared'}}
    }
  }

  void g() {
    (void)^(bool b){
      NonPOD np;
      if (b) {
        goto undeclared; // expected-error{{use of undeclared label 'undeclared'}}
      }
    };
  }
}

extern "C" {
  void exit(int);
}

void f() {
  {
    goto exit;
  }
 exit:
  return;
}

namespace PR10620 {
  struct S {
    ~S() {}
  };
  void g(const S& s) {
    goto done; // expected-error {{cannot jump}}
    const S s2(s); // expected-note {{jump bypasses variable initialization}}
  done:
    ;
  }
}

namespace test12 {
  struct A { A(); A(const A&); ~A(); };
  void test(A a) { // expected-note {{jump enters lifetime of block}} FIXME: weird location
    goto lbl; // expected-error {{cannot jump}}
    (void) ^{ (void) a; };
  lbl:
    return;
  }
}
