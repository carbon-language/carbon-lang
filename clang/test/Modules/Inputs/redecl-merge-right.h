__import_module__ redecl_merge_top;

@interface Super
@end

@interface A : Super
- (Super*)init;
@end

@class B;

@protocol P1
- (void)protoMethod1;
@end

@protocol P1;

@protocol P2;

@protocol P2;

@protocol P2;

// Test declarations in different modules with no common initial
// declaration.
@class C;
C *get_a_C(void);
@class C2;
C2 *get_a_C2(void);
@class C3;
C3 *get_a_C3(void);

@class C4;
@class C4;
@class C4;
@class C4;
C4 *get_a_C4(void);

@class Explicit;

int *explicit_func(void);

struct explicit_struct;

@protocol P4, P3;
@protocol P3;
@protocol P3;
@protocol P3;

#ifdef __cplusplus
template<typename T> class Vector { 
public:
  void push_back(const T&);
};
#endif

int ONE;
__import_module__ redecl_merge_top.Explicit;
const int one = ONE;
