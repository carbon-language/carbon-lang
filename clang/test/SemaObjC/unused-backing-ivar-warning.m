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

// rdar://15630719
@interface CDBModifyRecordsOperation : NSObject
@property (nonatomic, assign) BOOL atomic;
@end

@class NSString;

@implementation CDBModifyRecordsOperation
- (void)setAtomic:(BOOL)atomic {
  if (atomic == __objc_yes) {
    NSString *recordZoneID = 0;
  }
  _atomic = atomic;
}
@end

// rdar://15728901
@interface GATTOperation : NSObject {
    long operation;
}
@property(assign) long operation;
@end

@implementation GATTOperation
@synthesize operation;
+ (id) operation {
    return 0;
}
@end

// rdar://15727327
@interface Radar15727327 : NSObject
@property (assign, readonly) long p;
@property (assign) long q; // expected-note 2 {{property declared here}}
@property (assign, readonly) long r; // expected-note {{property declared here}}
- (long)Meth;
@end

@implementation Radar15727327
@synthesize p;
@synthesize q;
@synthesize r;
- (long)Meth { return p; }
- (long) p { [self Meth]; return 0;  }
- (long) q { return 0; } // expected-warning {{ivar 'q' which backs the property is not referenced in this property's accessor}}
- (void) setQ : (long) val { } // expected-warning {{ivar 'q' which backs the property is not referenced in this property's accessor}}
- (long) r { [self Meth]; return p; } // expected-warning {{ivar 'r' which backs the property is not referenced in this property's accessor}}
@end

