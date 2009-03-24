// RUN: clang-cc -fsyntax-only -verify %s


@interface Object
+ (id) new;
@end

@protocol GCObject
@property int class;
@end

@protocol DerivedGCObject <GCObject>
@property int Dclass;
@end

@interface GCObject  : Object <DerivedGCObject> {
    int ifield;
    int iOwnClass;
    int iDclass;
}
@property int OwnClass;
@end

@implementation GCObject : Object
@synthesize class=ifield;
@synthesize Dclass=iDclass;
@synthesize OwnClass=iOwnClass;
@end

int main(int argc, char **argv) {
    GCObject *f = [GCObject new];
    f.class = 5;
    f.Dclass = 1;
    f.OwnClass = 3;
    return f.class + f.Dclass  + f.OwnClass - 9;
}
