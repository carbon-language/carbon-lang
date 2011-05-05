// RUN: %clang_cc1 -emit-llvm-only -fblocks -o - -triple x86_64-apple-darwin10 %s
// rdar://9362021

@class DYFuture;
@interface NSCache
- (void)setObject:(id)obj forKey:(id)key;
@end

template <typename T>
class ResourceManager
{
public:
 ~ResourceManager();
 DYFuture* XXX();
 NSCache* _spDeviceCache;
};

template <typename T>
DYFuture* ResourceManager<T>::XXX()
{
 ^ {
   [_spDeviceCache setObject:0 forKey:0];
  }();

 return 0;
}

struct AnalyzerBaseObjectTypes { };

void FUNC()
{
    ResourceManager<AnalyzerBaseObjectTypes> *rm;
    ^(void) { rm->XXX(); }();
}
