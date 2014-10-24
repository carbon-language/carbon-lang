// RUN: %clang_cc1 -triple armv7-ios5.0 -std=c++11 -fobjc-arc -Os -emit-llvm -o - %s \
// RUN:     | FileCheck %s

typedef __SIZE_TYPE__ size_t;

namespace std {
template <typename _Ep>
class initializer_list {
  const _Ep* __begin_;
  size_t __size_;

  initializer_list(const _Ep* __b, size_t __s);
};
}

@interface I
+ (instancetype) new;
@end

void function(std::initializer_list<I *>);

extern "C" void single() { function({ [I new] }); }

// CHECK: [[INSTANCE:%.*]] = {{.*}} call {{.*}} i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}})
// CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[INSTANCE]] to %0*
// CHECK-NEXT: store %0* [[CAST]], %0** [[ARRAY:%.*]],
// CHECK: call {{.*}} void @objc_release(i8* {{.*}})

extern "C" void multiple() { function({ [I new], [I new] }); }

// CHECK: [[INSTANCE:%.*]] = {{.*}} call {{.*}} i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}})
// CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[INSTANCE]] to %0*
// CHECK-NEXT: store %0* [[CAST]], %0** [[ARRAY:%.*]],
// CHECK: call {{.*}} void @objc_release(i8* {{.*}})
// CHECK-NEXT: icmp eq

void external();

extern "C" void extended() {
  const auto && temporary = { [I new] };
  external();
}

// CHECK: [[INSTANCE:%.*]] = {{.*}} call {{.*}} i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}})
// CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[INSTANCE:%.*]] to {{.*}}*
// CHECK-NEXT: store {{.*}}* [[CAST]], {{.*}}** {{.*}}
// CHECK: {{.*}} call {{.*}} void @_Z8externalv()
// CHECK: {{.*}} call {{.*}} void @objc_release(i8* {{.*}})

std::initializer_list<I *> il = { [I new] };

// CHECK: [[POOL:%.*]] = {{.*}} call {{.*}} i8* @objc_autoreleasePoolPush()
// CHECK: [[INSTANCE:%.*]] = {{.*}} call {{.*}} i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}})
// CHECK-NEXT: [[CAST:%.*]] = bitcast i8* [[INSTANCE]] to %0*
// CHECK-NEXT: store %0* [[CAST]], %0** getelementptr inbounds ([1 x %0*]* @_ZGR2il_, i32 0, i32 0)
// CHECK: {{.*}} call {{.*}} void @objc_autoreleasePoolPop(i8* [[POOL]])

