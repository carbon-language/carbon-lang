// RUN: clang-cc -Wchar-subscripts -fsyntax-only -verify %s

void t1() {
  int array[1] = { 0 };
  char subscript = 0;
  int val = array[subscript]; // expected-warning{{array subscript is of type 'char'}}
}

void t2() {
  int array[1] = { 0 };
  char subscript = 0;
  int val = subscript[array]; // expected-warning{{array subscript is of type 'char'}}
}

void t3() {
  int *array = 0;
  char subscript = 0;
  int val = array[subscript]; // expected-warning{{array subscript is of type 'char'}}
}

void t4() {
  int *array = 0;
  char subscript = 0;
  int val = subscript[array]; // expected-warning{{array subscript is of type 'char'}}
}

char returnsChar();
void t5() {
  int *array = 0;
  int val = array[returnsChar()]; // expected-warning{{array subscript is of type 'char'}}
}
