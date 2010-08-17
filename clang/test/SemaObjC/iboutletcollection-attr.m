// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s
// rdar: // 8308053

@interface I {
    __attribute__((iboutletcollection(I))) id ivar1;
    __attribute__((iboutletcollection(id))) id ivar2;
    __attribute__((iboutletcollection())) id ivar3;
    __attribute__((iboutletcollection)) id ivar4;
}
@property (nonatomic, retain) __attribute__((iboutletcollection(I))) id prop1;
@property (nonatomic, retain) __attribute__((iboutletcollection(id))) id prop2;
@property (nonatomic, retain) __attribute__((iboutletcollection())) id prop3;
@property (nonatomic, retain) __attribute__((iboutletcollection)) id prop4;
@end

typedef void *PV;
@interface BAD {
    __attribute__((iboutletcollection(I, 1))) id ivar1; // expected-error {{attribute requires 1 argument(s)}}
    __attribute__((iboutletcollection(B))) id ivar2; // expected-error {{invalid type 'B' as argument of iboutletcollection attribue}}
    __attribute__((iboutletcollection(PV))) id ivar3; // expected-error {{invalid type 'PV' as argument of iboutletcollection attribue}}
    __attribute__((iboutletcollection(PV))) void *ivar4; // expected-error {{ivar with iboutletcollection attribue must have object type (invalid 'void *')}}
}
@property (nonatomic, retain) __attribute__((iboutletcollection(I,2,3))) id prop1; // expected-error {{attribute requires 1 argument(s)}}
@property (nonatomic, retain) __attribute__((iboutletcollection(B))) id prop2; // expected-error {{invalid type 'B' as argument of iboutletcollection attribue}}

@property __attribute__((iboutletcollection(BAD))) int prop3; // expected-error {{property with iboutletcollection attribue must have object type (invalid 'int')}}
@end

