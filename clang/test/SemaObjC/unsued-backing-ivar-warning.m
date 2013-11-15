// RUN: %clang_cc1  -fsyntax-only -Wunused-property-ivar -verify -Wno-objc-root-class %s
// rdar://14989999

@interface NSObject @end

@interface Example : NSObject
@property (nonatomic, copy) id t; // expected-note {{property declared here}}
@property (nonatomic, copy) id u; // expected-note {{property declared here}}
@property (nonatomic, copy) id v; // expected-note {{property declared here}}
@property (nonatomic, copy) id w;
@property (nonatomic, copy) id x; // expected-note {{property declared here}}
@property (nonatomic, copy) id y; // expected-note {{property declared here}}
@property (nonatomic, copy) id z;
@property (nonatomic, copy) id ok;
@end

@implementation Example
- (void) setX:(id)newX {  // expected-warning {{ivar '_x' which backs the property is not referenced in this property's accessor}}
    _y = newX;
}
- (id) y { // expected-warning {{ivar '_y' which backs the property is not referenced in this property's accessor}}
  return _v;
}

- (void) setV:(id)newV { // expected-warning {{ivar '_v' which backs the property is not referenced in this property's accessor}}
    _y = newV;
}

// No warning here because there is no backing ivar.
// both setter/getter are user defined.
- (void) setW:(id)newW {
    _y = newW;
}
- (id) w {
  return _v;
}

- (id) u { // expected-warning {{ivar '_u' which backs the property is not referenced in this property's accessor}}
  return _v;
}

@synthesize ok = okIvar;
- (void) setOk:(id)newOk {
    okIvar = newOk;
}

@synthesize t = tIvar;
- (void) setT:(id)newT { // expected-warning {{ivar 'tIvar' which backs the property is not referenced in this property's accessor}}
    okIvar = newT;
}
@end

// rdar://15473432
typedef char BOOL;
@interface CalDAVServerVersion {
  BOOL _supportsTimeRangeFilterWithoutEndDate;
}
@property (nonatomic, readonly,nonatomic) BOOL supportsTimeRangeFilterWithoutEndDate;
@end

@interface CalDAVConcreteServerVersion : CalDAVServerVersion {
}
@end

@interface CalendarServerVersion : CalDAVConcreteServerVersion
@end

@implementation CalDAVServerVersion
@synthesize supportsTimeRangeFilterWithoutEndDate=_supportsTimeRangeFilterWithoutEndDate;
@end

@implementation CalendarServerVersion
-(BOOL)supportsTimeRangeFilterWithoutEndDate {
  return 0;
}
@end
