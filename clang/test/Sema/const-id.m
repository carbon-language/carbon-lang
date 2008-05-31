// RUN: clang %s -verify -fsyntax-only

int main() {
    const id foo;
    [foo bar];  // expected-warning {{method '-bar' not found (return type defaults to 'id')}}
    return 0;
}

