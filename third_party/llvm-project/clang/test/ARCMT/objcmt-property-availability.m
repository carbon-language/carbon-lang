// RUN: rm -rf %t
// RUN: %clang_cc1  -objcmt-migrate-readwrite-property -objcmt-migrate-readonly-property -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc %s.result
// rdar://15300059


#define __NSi_7_0 introduced=7.0
#define __NSi_6_0 introduced=6.0

#define CF_AVAILABLE(_mac, _ios) __attribute__((availability(ios,__NSi_##_ios)))
#define CF_AVAILABLE_MAC(_mac) __attribute__((availability(macosx,__NSi_##_mac)))
#define CF_AVAILABLE_IOS(_ios) __attribute__((availability(macosx,unavailable)))

#define NS_AVAILABLE(_mac, _ios) CF_AVAILABLE(_mac, _ios)
#define NS_AVAILABLE_MAC(_mac) CF_AVAILABLE_MAC(_mac)
#define NS_AVAILABLE_IOS(_ios) CF_AVAILABLE_IOS(_ios)

#define UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))

@interface MKMapItem
- (MKMapItem *)source NS_AVAILABLE(10_9, 6_0);
- (void)setSource:(MKMapItem *)source NS_AVAILABLE(10_9, 7_0);

- (void)setDest:(MKMapItem *)source NS_AVAILABLE(10_9, 6_0);
- (MKMapItem *)dest NS_AVAILABLE(10_9, 6_0);

- (MKMapItem *)final;
- (void)setFinal:(MKMapItem *)source;

- (MKMapItem *)total NS_AVAILABLE(10_9, 6_0);
- (void)setTotal:(MKMapItem *)source;

- (MKMapItem *)comp NS_AVAILABLE(10_9, 6_0);
- (void)setComp:(MKMapItem *)source UNAVAILABLE;

- (MKMapItem *)tally  UNAVAILABLE NS_AVAILABLE(10_9, 6_0);
- (void)setTally:(MKMapItem *)source UNAVAILABLE NS_AVAILABLE(10_9, 6_0);

- (MKMapItem *)itally  NS_AVAILABLE(10_9, 6_0);
- (void)setItally:(MKMapItem *)source UNAVAILABLE NS_AVAILABLE(10_9, 6_0);

- (MKMapItem *)normal  UNAVAILABLE;
- (void)setNormal:(MKMapItem *)source UNAVAILABLE NS_AVAILABLE(10_9, 6_0);
@end

