// TLS variable cannot be aligned to more than 32 bytes on PS4.

// RUN: %clang_cc1 -triple x86_64-scei-ps4 -fsyntax-only -verify %s


// A non-aligned type.
struct non_aligned_struct {
    int some_data[16]; // 64 bytes of stuff, non aligned.
};

// An aligned type.
struct __attribute__(( aligned(64) )) aligned_struct {
    int some_data[12]; // 48 bytes of stuff, aligned to 64.
};

// A type with an aligned field.
struct  struct_with_aligned_field {
    int some_aligned_data[12] __attribute__(( aligned(64) )); // 48 bytes of stuff, aligned to 64.
};

// A typedef of the aligned struct.
typedef aligned_struct another_aligned_struct;

// A typedef to redefine a non-aligned struct as aligned.
typedef __attribute__(( aligned(64) )) non_aligned_struct yet_another_aligned_struct;

// Non aligned variable doesn't cause an error.
__thread non_aligned_struct foo;

// Variable aligned because of its type should cause an error.
__thread aligned_struct                    bar; // expected-error{{alignment (64) of thread-local variable}}

// Variable explicitly aligned in the declaration should cause an error.
__thread non_aligned_struct                bar2 __attribute__(( aligned(64) )); // expected-error{{alignment (64) of thread-local variable}}

// Variable aligned because of one of its fields should cause an error.
__thread struct_with_aligned_field         bar3; // expected-error{{alignment (64) of thread-local variable}}

// Variable aligned because of typedef, first case.
__thread another_aligned_struct            bar4; // expected-error{{alignment (64) of thread-local variable}}

// Variable aligned because of typedef, second case.
__thread yet_another_aligned_struct        bar5; // expected-error{{alignment (64) of thread-local variable}}

int baz ()
{
    return foo.some_data[0] + bar.some_data[1] + bar2.some_data[2] +
           bar3.some_aligned_data[3] + bar4.some_data[4] +
           bar5.some_data[5];
}


// Verify alignment check where a dependent type is involved.
// The check is (correctly) not performed on "t", but the check still is
// performed on the structure as a whole once it has been instantiated.

template<class T> struct templated_tls {
    static __thread T t;
    T other_t __attribute__(( aligned(64) ));
};
__thread templated_tls<int> blah; // expected-error{{alignment (64) of thread-local variable}}

int blag() {
    return blah.other_t * 2;
}


// Verify alignment check where the alignment is a template parameter.
// The check is only performed during instantiation.
template <int N>
struct S {
  static int __thread __attribute__((aligned(N))) x; // expected-error{{alignment (64) of thread-local variable}}
};

S<64> s_instance; // expected-note{{in instantiation of template class 'S<64>' requested here}}
