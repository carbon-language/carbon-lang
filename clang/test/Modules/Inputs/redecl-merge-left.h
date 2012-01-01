__import_module__ redecl_merge_top;

@class A;

@class A;

@interface B
+ (B*) create_a_B;
@end

@class A;

@protocol P1;
@protocol P2
- (void)protoMethod2;
@end

// Test declarations in different modules with no common initial
// declaration.
@class C;
void accept_a_C(C*);

@class C2;
void accept_a_C2(C2*);

@class C3;
void accept_a_C3(C3*);
@class C3;

@class C4;

@class Explicit;

int *explicit_func(void);

struct explicit_struct;

@protocol P3, P4;

@protocol P3;

#ifdef __cplusplus
template<typename T> class Vector;

template<typename T> class Vector;
#endif
