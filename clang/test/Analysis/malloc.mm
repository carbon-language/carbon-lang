// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-store=region -verify %s
#include "system-header-simulator-objc.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

// Done with headers. Start testing.
void testNSDatafFreeWhenDoneNoError(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [NSData dataWithBytesNoCopy:data length:dataLength];
  free(data); // no warning
}

void testNSDataFreeWhenDoneYES(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [NSData dataWithBytesNoCopy:data length:dataLength freeWhenDone:1]; // no-warning
}

void testNSDataFreeWhenDoneYES2(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [[NSData alloc] initWithBytesNoCopy:data length:dataLength freeWhenDone:1]; // no-warning
}


void testNSStringFreeWhenDoneYES(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithBytesNoCopy:data length:dataLength encoding:NSUTF8StringEncoding freeWhenDone:1]; // no-warning
}

void testNSStringFreeWhenDoneYES2(NSUInteger dataLength) {
  unichar *data = (unichar*)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithCharactersNoCopy:data length:dataLength freeWhenDone:1]; // no-warning
}


void testNSDataFreeWhenDoneNO(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [NSData dataWithBytesNoCopy:data length:dataLength freeWhenDone:0]; // expected-warning{{leak}}
}

void testNSDataFreeWhenDoneNO2(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [[NSData alloc] initWithBytesNoCopy:data length:dataLength freeWhenDone:0]; // expected-warning{{leak}}
}


void testNSStringFreeWhenDoneNO(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithBytesNoCopy:data length:dataLength encoding:NSUTF8StringEncoding freeWhenDone:0]; // expected-warning{{leak}}
}

void testNSStringFreeWhenDoneNO2(NSUInteger dataLength) {
  unichar *data = (unichar*)malloc(42);
  NSString *nsstr = [[NSString alloc] initWithCharactersNoCopy:data length:dataLength freeWhenDone:0]; // expected-warning{{leak}}
}

// TODO: False Negative.
void testNSDatafFreeWhenDoneFN(NSUInteger dataLength) {
  unsigned char *data = (unsigned char *)malloc(42);
  NSData *nsdata = [NSData dataWithBytesNoCopy:data length:dataLength freeWhenDone:1];
  free(data); // false negative
}
