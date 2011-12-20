__import_module__ redecl_merge_top;

@class A;

@class A;

@interface B
+ (B*) create_a_B;
@end

@class A;

@class Explicit;

int *explicit_func(void);

struct explicit_struct;

#ifdef __cplusplus
template<typename T> class Vector;

template<typename T> class Vector;
#endif
