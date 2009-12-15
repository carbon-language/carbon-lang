// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface INTF
- (void) meth;
- (void) meth : (int) arg1;
- (int)  int_meth; 
+ (int) cls_meth;
+ (void) cls_meth1 : (int) arg1; 
@end

@implementation INTF 	// expected-warning {{incomplete implementation}} expected-warning {{method definition for 'int_meth' not found}} expected-warning {{method definition for 'cls_meth' not found}} expected-warning {{method definition for 'cls_meth1:' not found}}
- (void) meth {}
- (void) meth : (int) arg2{}
- (void) cls_meth1 : (int) arg2{}
@end

@interface INTF1
- (void) meth;
- (void) meth : (int) arg1;
- (int)  int_meth;      
+ (int) cls_meth;       
+ (void) cls_meth1 : (int) arg1; 
@end

@implementation INTF1 // expected-warning {{incomplete implementation}} expected-warning {{method definition for 'int_meth' not found}} expected-warning {{method definition for 'cls_meth' not found}} expected-warning {{method definition for 'cls_meth1:' not found}}
- (void) meth {}
- (void) meth : (int) arg2{}
- (void) cls_meth1 : (int) arg2{}
@end

@interface INTF2
- (void) meth;
- (void) meth : (int) arg1;
- (void) cls_meth1 : (int) arg1; 
@end

@implementation INTF2
- (void) meth {}
- (void) meth : (int) arg2{}
- (void) cls_meth1 : (int) arg2{}
@end

