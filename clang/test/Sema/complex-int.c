// RUN: clang %s -verify -fsyntax-only

void a() {
__complex__ int arr;
__complex__ short brr;
__complex__ int result;

result = arr*brr;
}

