// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.osx.cocoa.InstanceVariableInvalidation -fobjc-default-synthesize-properties -verify %s

@protocol NSObject
@end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

@protocol Invalidation1 <NSObject> 
- (void) invalidate __attribute__((annotate("objc_instance_variable_invalidator")));
@end 

@protocol Invalidation2 <NSObject> 
- (void) invalidate __attribute__((annotate("objc_instance_variable_invalidator")));
@end 

@protocol Invalidation3 <NSObject>
- (void) invalidate __attribute__((annotate("objc_instance_variable_invalidator")));
@end

@interface Invalidation2Class <Invalidation2>
@end

@interface Invalidation1Class <Invalidation1>
@end

@interface SomeInvalidationImplementingObject: NSObject <Invalidation3, Invalidation2> {
  SomeInvalidationImplementingObject *ObjA;
}
@end

@implementation SomeInvalidationImplementingObject
- (void)invalidate{
  ObjA = 0;
}
@end

@interface SomeSubclassInvalidatableObject : SomeInvalidationImplementingObject {
  SomeInvalidationImplementingObject *Obj1; // expected-warning{{Ivar needs to be invalidated}}
  SomeInvalidationImplementingObject *Obj2; // expected-warning{{Ivar needs to be invalidated}}
  SomeInvalidationImplementingObject *Obj3;
  SomeInvalidationImplementingObject *_Prop1;
  SomeInvalidationImplementingObject *_Prop4;
  SomeInvalidationImplementingObject *_propIvar;
  Invalidation1Class *MultipleProtocols; // expected-warning{{Ivar needs to be invalidated}}
  Invalidation2Class *MultInheritance; // expected-warning{{Ivar needs to be invalidated}}

  // No warnings on these.
  NSObject *NObj1;
  NSObject *NObj2;
  NSObject *_NProp1;
  NSObject *_NpropIvar;
}

@property (assign) SomeInvalidationImplementingObject* Prop0;
@property (nonatomic, assign) SomeInvalidationImplementingObject* Prop1;
@property (assign) SomeInvalidationImplementingObject* Prop2;
@property (assign) SomeInvalidationImplementingObject* Prop3;
@property (assign) SomeInvalidationImplementingObject* Prop4;
@property (assign) NSObject* NProp0;
@property (nonatomic, assign) NSObject* NProp1;
@property (assign) NSObject* NProp2;

-(void)setProp1: (SomeInvalidationImplementingObject*) InO;
-(void)setNProp1: (NSObject*) InO;

-(void)invalidate;

@end

@implementation SomeSubclassInvalidatableObject

@synthesize Prop2 = _propIvar;
@synthesize Prop3;

- (void) setProp1: (SomeInvalidationImplementingObject*) InObj {
  _Prop1 = InObj;
}

- (void) setProp4: (SomeInvalidationImplementingObject*) InObj {
  _Prop4 = InObj;
}
- (SomeInvalidationImplementingObject*) Prop4 {
  return _Prop4;
}

@synthesize NProp2 = _NpropIvar;

- (void) setNProp1: (NSObject*) InObj {
  _NProp1 = InObj;
}

- (void) invalidate {
   [Obj3 invalidate];
   self.Prop1 = 0;
   [self setProp2:0];
   [self setProp3:0];
   self.Prop4 = 0;
   [super invalidate];
}
@end
