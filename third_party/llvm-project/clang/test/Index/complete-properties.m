/* Note: the RUN lines are near the end of the file, since line/column
 matter for this test. */
@class MyClass;
@interface I1 
{
  id StoredProp3;
  int RandomIVar;
}
@property int Prop0;
@property int Prop1;
@property float Prop2;
@end

@interface I2 : I1
@property id Prop3;
@property id Prop4;
@end

@implementation I2
@synthesize Prop2, Prop1, Prop3 = StoredProp3;
@dynamic Prop4;
@end

@interface I3 : I2
@property id Prop3;
@end

id test(I3 *i3) {
  return i3.Prop3;
}

@interface I4
@property id Prop2;
@end

@interface I4 () {
  I4 *Prop1;
}
@end

@implementation I4 {
  id Prop2_;
}

@synthesize Prop2 = Prop2_;
@end

@protocol P1
@property int Prop5;
@end

@class P1;

@interface I5<P1>
@end
@implementation I5
@synthesize Prop5;
@end
// RUN: c-index-test -code-completion-at=%s:20:13 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCPropertyDecl:{ResultType int}{TypedText Prop0}
// CHECK-CC1: ObjCPropertyDecl:{ResultType int}{TypedText Prop1}
// CHECK-CC1: ObjCPropertyDecl:{ResultType float}{TypedText Prop2}
// CHECK-CC1: ObjCPropertyDecl:{ResultType id}{TypedText Prop3}
// CHECK-CC1: ObjCPropertyDecl:{ResultType id}{TypedText Prop4}
// RUN: c-index-test -code-completion-at=%s:20:20 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCPropertyDecl:{ResultType int}{TypedText Prop0}
// CHECK-CC2: ObjCPropertyDecl:{ResultType int}{TypedText Prop1}
// CHECK-CC2-NEXT: ObjCPropertyDecl:{ResultType id}{TypedText Prop3}
// CHECK-CC2: ObjCPropertyDecl:{ResultType id}{TypedText Prop4}
// RUN: c-index-test -code-completion-at=%s:20:35 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCIvarDecl:{ResultType id}{TypedText _Prop3} (36)
// CHECK-CC3: ObjCIvarDecl:{ResultType int}{TypedText RandomIVar} (35)
// CHECK-CC3: ObjCIvarDecl:{ResultType id}{TypedText StoredProp3} (8)

// RUN: c-index-test -code-completion-at=%s:21:10 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCPropertyDecl:{ResultType int}{TypedText Prop0}
// CHECK-CC4-NEXT: ObjCPropertyDecl:{ResultType id}{TypedText Prop4}

// RUN: c-index-test -code-completion-at=%s:29:13 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCPropertyDecl:{ResultType int}{TypedText Prop0} (37)
// CHECK-CC5-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText Prop1} (37)
// CHECK-CC5-NEXT: ObjCPropertyDecl:{ResultType float}{TypedText Prop2} (37)
// CHECK-CC5-NEXT: ObjCPropertyDecl:{ResultType id}{TypedText Prop3} (35)
// CHECK-CC5-NEXT: ObjCPropertyDecl:{ResultType id}{TypedText Prop4} (37)

// RUN: c-index-test -code-completion-at=%s:9:11 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInterfaceDecl:{TypedText MyClass} (50)


// RUN: c-index-test -code-completion-at=%s:45:21 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7-NOT: ObjCIvarDecl:{ResultType id}{TypedText _Prop2}
// CHECK-CC7: ObjCIvarDecl:{ResultType I4 *}{TypedText Prop1} (17)
// CHECK-CC7: ObjCIvarDecl:{ResultType id}{TypedText Prop2_} (7)

// RUN: c-index-test -code-completion-at=%s:57:13 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: ObjCPropertyDecl:{ResultType int}{TypedText Prop5} (37)

@interface ClassProperties

@property int instanceProperty;
@property(class) int explicit;
@property(class, readonly) int explicitReadonly;

+ (int)implicit;
+ (int)setImplicit:(int)x;

+ (int)implicitReadonly;

+ (void)noProperty;

- (int)implicitInstance;

+ (int)shadowedImplicit;

@end

@interface ClassProperties (Category)

+ (int)implicitInCategory;

@end

@protocol ProtocolClassProperties

@property(class, readonly) int explicitInProtocol;

@end

@interface SubClassProperties: ClassProperties <ProtocolClassProperties>

@property(class) ClassProperties *shadowedImplicit;

@end

@implementation SubClassProperties

-(void) foo {
  super.instanceProperty;
}

@end

void classProperties() {
  (void)ClassProperties.implicit;
  (void)SubClassProperties.explicit;
}

// RUN: c-index-test -code-completion-at=%s:144:25 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-CC9 %s
// CHECK-CC9: ObjCPropertyDecl:{ResultType int}{TypedText explicit} (35)
// CHECK-CC9-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText explicitReadonly} (35)
// CHECK-CC9-NEXT: ObjCClassMethodDecl:{ResultType int}{TypedText implicit} (37)
// CHECK-CC9-NEXT: ObjCClassMethodDecl:{ResultType int}{TypedText implicitInCategory} (37)
// CHECK-CC9-NEXT: ObjCClassMethodDecl:{ResultType int}{TypedText implicitReadonly} (37)
// CHECK-CC9-NEXT: ObjCClassMethodDecl:{ResultType int}{TypedText shadowedImplicit} (37)
// CHECK-CC9-NOT: implicitInstance
// CHECK-CC9-NOT: noProperty
// CHECK-CC9-NOT: instanceProperty

// RUN: c-index-test -code-completion-at=%s:145:28 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-CC10 %s
// CHECK-CC10: ObjCPropertyDecl:{ResultType int}{TypedText explicit} (37)
// CHECK-CC10-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText explicitInProtocol} (37)
// CHECK-CC10-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText explicitReadonly} (37)
// CHECK-CC10-NEXT: ObjCClassMethodDecl:{ResultType int}{TypedText implicit} (39)
// CHECK-CC10-NEXT: ObjCClassMethodDecl:{ResultType int}{TypedText implicitInCategory} (39)
// CHECK-CC10-NEXT: ObjCClassMethodDecl:{ResultType int}{TypedText implicitReadonly} (39)
// CHECK-CC10-NEXT: ObjCPropertyDecl:{ResultType ClassProperties *}{TypedText shadowedImplicit} (35)
// CHECK-CC10-NOT: implicitInstance
// CHECK-CC10-NOT: noProperty
// CHECK-CC10-NOT: instanceProperty

// RUN: c-index-test -code-completion-at=%s:138:9 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-CC11 %s
// CHECK-CC11-NOT: explicit
// CHECK-CC11-NOT: explicitReadonly
// CHECK-CC11-NOT: implicit
// CHECK-CC11-NOT: implicitReadonly
// CHECK-CC11-NOT: shadowedImplicit
// CHECK-CC11-NOT: implicitInCategory
