// RUN: clang -checker-cfref -verify %s

typedef signed long CFIndex;
typedef const struct __CFAllocator * CFAllocatorRef;
enum { kCFNumberSInt8Type = 1, kCFNumberSInt16Type = 2,
       kCFNumberSInt32Type = 3, kCFNumberSInt64Type = 4,
       kCFNumberFloat32Type = 5, kCFNumberFloat64Type = 6,
       kCFNumberCharType = 7, kCFNumberShortType = 8,
       kCFNumberIntType = 9, kCFNumberLongType = 10,
       kCFNumberLongLongType = 11, kCFNumberFloatType = 12,
       kCFNumberDoubleType = 13, kCFNumberCFIndexType = 14,
       kCFNumberNSIntegerType = 15, kCFNumberCGFloatType = 16,
       kCFNumberMaxType = 16 };
typedef CFIndex CFNumberType;
typedef const struct __CFNumber * CFNumberRef;
extern CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);

#include <stdint.h>

CFNumberRef f1() {
  uint8_t x = 1;
  return CFNumberCreate(0, kCFNumberSInt16Type, &x);  // expected-warning{{An 8 bit integer is used to initialize a CFNumber object that represents a 16 bit integer. 8 bits of the CFNumber value will be garbage.}}
}

CFNumberRef f2() {
  uint16_t x = 1;
  return CFNumberCreate(0, kCFNumberSInt8Type, &x); // expected-warning{{A 16 bit integer is used to initialize a CFNumber object that represents an 8 bit integer. 8 bits of the input integer will be lost.}}
}
