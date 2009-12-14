// RUN: clang -cc1 -fsyntax-only -Wreadonly-setter-attrs -verify %s  -fblocks

// Check property attribute consistency.

@interface I0
@property(readonly, readwrite) int p0; // expected-error {{property attributes 'readonly' and 'readwrite' are mutually exclusive}}

@property(retain) int p1; // expected-error {{property with 'retain' attribute must be of object type}}

@property(copy) int p2; // expected-error {{property with 'copy' attribute must be of object type}}

@property(assign, copy) id p3_0; // expected-error {{property attributes 'assign' and 'copy' are mutually exclusive}} 
@property(assign, retain) id p3_1; // expected-error {{property attributes 'assign' and 'retain' are mutually exclusive}} 
@property(copy, retain) id p3_2; // expected-error {{property attributes 'copy' and 'retain' are mutually exclusive}} 
@property(assign, copy, retain) id p3_3; // expected-error {{property attributes 'assign' and 'copy' are mutually exclusive}}, expected-error {{property attributes 'assign' and 'retain' are mutually exclusive}} 

@property id p4; // expected-warning {{no 'assign', 'retain', or 'copy' attribute is specified - 'assign' is assumed}}, expected-warning {{default property attribute 'assign' not appropriate for non-gc object}}

@property(nonatomic,copy) int (^includeMailboxCondition)(); 
@property(nonatomic,copy) int (*includeMailboxCondition2)(); // expected-error {{property with 'copy' attribute must be of object type}}

@end
