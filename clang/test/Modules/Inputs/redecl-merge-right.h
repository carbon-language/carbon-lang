__import_module__ redecl_merge_top;

@interface Super
@end

@interface A : Super
- (Super*)init;
@end

@class B;

#ifdef __cplusplus
template<typename T> class Vector { 
public:
  void push_back(const T&);
};
#endif
