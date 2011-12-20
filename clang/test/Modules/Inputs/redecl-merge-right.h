__import_module__ redecl_merge_top;

@interface Super
@end

@interface A : Super
- (Super*)init;
@end

@class B;

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
