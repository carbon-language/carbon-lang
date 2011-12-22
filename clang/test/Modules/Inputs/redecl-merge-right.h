__import_module__ redecl_merge_top;

@interface Super
@end

@interface A : Super
- (Super*)init;
@end

@class B;

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

#ifdef __cplusplus
template<typename T> class Vector { 
public:
  void push_back(const T&);
};
#endif

int ONE;
__import_module__ redecl_merge_top.Explicit;
const int one = ONE;
