// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=core -analyzer-output=text -verify %s

typedef unsigned char uint8_t;

#define UINT8_MAX 255
#define TCP_MAXWIN 65535

uint8_t get_uint8_max() {
  uint8_t rcvscale = UINT8_MAX; // expected-note{{'rcvscale' initialized to 255}}
  return rcvscale; // expected-note{{Returning the value 255 (loaded from 'rcvscale')}}
}

void shift_by_undefined_value() {
  uint8_t shift_amount = get_uint8_max(); // expected-note{{'shift_amount' initialized to 255}}
                                // expected-note@-1{{Calling 'get_uint8_max'}}
                                // expected-note@-2{{Returning from 'get_uint8_max'}}
  (void)(TCP_MAXWIN << shift_amount); // expected-warning{{The result of the left shift is undefined due to shifting by '255', which is greater or equal to the width of type 'int'}}
                                      // expected-note@-1{{The result of the left shift is undefined due to shifting by '255', which is greater or equal to the width of type 'int'}}
}

namespace array_index_tracking {
void consume(int);

int getIndex(int x) {
  int a;
  if (x > 0)
    a = 3;
  else
    a = 2;
  return a;
}

int getInt();

void testArrayIndexTracking() {
  int arr[10];

  for (int i = 0; i < 3; ++i)
    arr[i] = 0;
  int x = getInt();
  int n = getIndex(x);
  consume(arr[n]);
}
} // end of namespace array_index_tracking
