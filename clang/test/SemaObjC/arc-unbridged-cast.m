// // RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s

typedef const struct __CFString * CFStringRef;

@interface Object
@property CFStringRef property;
- (CFStringRef) implicitProperty;
- (CFStringRef) newString;
- (CFStringRef) makeString;
@end

extern Object *object;

// rdar://9744349
id test0(void) {
  id p1 = (id)[object property];
  id p2 = (__bridge_transfer id)[object property];
  id p3 = (__bridge id)[object property];
  return (id) object.property;
}

// rdar://10140692
CFStringRef unauditedString(void);
CFStringRef plusOneString(void) __attribute__((cf_returns_retained));

#pragma clang arc_cf_code_audited begin
CFStringRef auditedString(void);
CFStringRef auditedCreateString(void);
#pragma clang arc_cf_code_audited end

void test1(int cond) {
  id x;
  x = (id) auditedString();
  x = (id) (cond ? auditedString() : (void*) 0);
  x = (id) (cond ? (void*) 0 : auditedString());
  x = (id) (cond ? (CFStringRef) @"help" : auditedString());

  x = (id) unauditedString(); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use __bridge_transfer to}}
  x = (id) (cond ? unauditedString() : (void*) 0); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use __bridge_transfer to}}
  x = (id) (cond ? (void*) 0 : unauditedString()); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use __bridge_transfer to}}
  x = (id) (cond ? (CFStringRef) @"help" : unauditedString()); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use __bridge_transfer to}}

  x = (id) auditedCreateString(); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use __bridge_transfer to}}
  x = (id) (cond ? auditedCreateString() : (void*) 0); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use __bridge_transfer to}}
  x = (id) (cond ? (void*) 0 : auditedCreateString()); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use __bridge_transfer to}}
  x = (id) (cond ? (CFStringRef) @"help" : auditedCreateString()); // expected-error {{requires a bridged cast}} expected-note {{use __bridge to}} expected-note {{use __bridge_transfer to}}

  x = (id) [object property];
  x = (id) (cond ? [object property] : (void*) 0);
  x = (id) (cond ? (void*) 0 : [object property]);
  x = (id) (cond ? (CFStringRef) @"help" : [object property]);  

  x = (id) object.property;
  x = (id) (cond ? object.property : (void*) 0);
  x = (id) (cond ? (void*) 0 : object.property);
  x = (id) (cond ? (CFStringRef) @"help" : object.property);  

  x = (id) object.implicitProperty;
  x = (id) (cond ? object.implicitProperty : (void*) 0);
  x = (id) (cond ? (void*) 0 : object.implicitProperty);
  x = (id) (cond ? (CFStringRef) @"help" : object.implicitProperty);  

  x = (id) [object makeString];
  x = (id) (cond ? [object makeString] : (void*) 0);
  x = (id) (cond ? (void*) 0 : [object makeString]);
  x = (id) (cond ? (CFStringRef) @"help" : [object makeString]);  

  x = (id) [object newString];
  x = (id) (cond ? [object newString] : (void*) 0);
  x = (id) (cond ? (void*) 0 : [object newString]);
  x = (id) (cond ? (CFStringRef) @"help" : [object newString]); // a bit questionable
}
