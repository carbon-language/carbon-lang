// RUN: %clang_cc1 -triple i386-apple-macosx10.6.6 -emit-llvm -fobjc-exceptions -fcxx-exceptions -fexceptions -o - %s | FileCheck %s
// rdar://8940528

@interface ns_array
+ (id) array;
@end

@implementation ns_array
+ (id) array { return 0; }
@end

id Groups();

id FUNC() {
    id groups;
    try
    {
      groups = Groups();  // throws on errors.
    }
    catch( id error )
    { 
      // CHECK: call i32 (i8*, i8*, ...)* @llvm.eh.selector({{.*}} @__gxx_personality_v0 {{.*}} @_ZTIP11objc_object
      error = error; 
      groups = [ns_array array]; 
    }
    return groups;

}

int main() {
  FUNC();
  return 0;
}
