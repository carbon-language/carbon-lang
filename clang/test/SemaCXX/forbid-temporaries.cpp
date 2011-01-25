// RUN: %clang_cc1 -fsyntax-only -verify %s

#if !__has_attribute(forbid_temporaries)
#error "Should support forbid_temporaries attribute"
#endif

class __attribute__((forbid_temporaries)) NotATemporary {
};

class __attribute__((forbid_temporaries(1))) ShouldntHaveArguments {  // expected-error {{attribute requires 0 argument(s)}}
};

void bad_function() __attribute__((forbid_temporaries));  // expected-warning {{'forbid_temporaries' attribute only applies to classes}}

int var __attribute__((forbid_temporaries));  // expected-warning {{'forbid_temporaries' attribute only applies to classes}}

void bar(const NotATemporary&);

void foo() {
  NotATemporary this_is_fine;
  bar(NotATemporary());  // expected-warning {{must not create temporaries of type 'NotATemporary'}}
  NotATemporary();   // expected-warning {{must not create temporaries of type 'NotATemporary'}}
}


// Check that the above restrictions work for templates too.
template<typename T>
class __attribute__((forbid_temporaries)) NotATemporaryTpl {
};

template<typename T>
void bar_tpl(const NotATemporaryTpl<T>&);

void tpl_user() {
  NotATemporaryTpl<int> this_is_fine;
  bar_tpl(NotATemporaryTpl<int>());  // expected-warning {{must not create temporaries of type 'NotATemporaryTpl<int>'}}
  NotATemporaryTpl<int>();   // expected-warning {{must not create temporaries of type 'NotATemporaryTpl<int>'}}
}


// Test that a specialization can override the template's default.
struct TemporariesOk;
template<> class NotATemporaryTpl<TemporariesOk> {
};

void specialization_user() {
  NotATemporaryTpl<TemporariesOk> this_is_fine;
  bar_tpl(NotATemporaryTpl<TemporariesOk>());
  NotATemporaryTpl<TemporariesOk>();
}
