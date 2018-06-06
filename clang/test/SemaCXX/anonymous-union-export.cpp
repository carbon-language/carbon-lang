// RUN: %clang_cc1 -std=c++17 -fmodules-ts -emit-obj -verify %s

export module M;
export {
    union { bool a; }; // expected-error{{anonymous unions at namespace or global scope must be declared 'static'}}
}
