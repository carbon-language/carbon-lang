// RUN: %clang_cc1  -fsyntax-only -verify %s

typedef unsigned int size_t;
@protocol P @end

@interface NSMutableArray
@end

@interface XNSMutableArray
@end

int main(void) {
id array;
id oldObject = array[10]; // expected-warning {{instance method '-objectAtIndexedSubscript:' not found (return type defaults to 'id')}}

array[10] = 0; // expected-warning {{instance method '-setObject:atIndexedSubscript:' not found (return type defaults to 'id')}}

id<P> p_array;
oldObject = p_array[10]; // expected-error {{expected method to read array element not found on object of type 'id<P>'}}

p_array[10] = 0; // expected-error {{expected method to write array element not found on object of type 'id<P>'}}
}
