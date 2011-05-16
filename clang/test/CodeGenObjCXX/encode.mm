// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: v17@0:8{vector<float, float, float>=}16
// CHECK: {vector<float, float, float>=}
// CHECK: v24@0:816

template <typename T1, typename T2, typename T3> struct vector {
  vector();
  vector(T1,T2,T3);
};

typedef vector< float, float, float > vector3f;

@interface SceneNode
{
 vector3f position;
}

@property (assign, nonatomic) vector3f position;

@end

@interface MyOpenGLView
{
@public
  vector3f position;
}
@property vector3f position;
@end

@implementation MyOpenGLView

@synthesize position;

-(void)awakeFromNib {
 SceneNode *sn;
 vector3f VF3(1.0, 1.0, 1.0);
 [sn setPosition:VF3];
}
@end


class Int3 { int x, y, z; };

// Enforce @encoding for member pointers.
@interface MemPtr {}
- (void) foo: (int (Int3::*)) member;
@end
@implementation MemPtr
- (void) foo: (int (Int3::*)) member {
}
@end

// rdar: // 8519948
typedef float HGVec4f __attribute__ ((vector_size(16)));

@interface RedBalloonHGXFormWrapper {
  HGVec4f m_Transform[4];
}
@end

@implementation RedBalloonHGXFormWrapper
@end

