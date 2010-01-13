/* Run lines are at the end, since line/column matter in this test. */
@interface MyClass { @public }
@end

@implementation MyClass
@end

// RUN: c-index-test -code-completion-at=%s:2:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText class}{HorizontalSpace  }{Placeholder identifier}{SemiColon ;}
// CHECK-CC1: {TypedText compatibility_alias}{HorizontalSpace  }{Placeholder alias}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText implementation}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText interface}{HorizontalSpace  }{Placeholder class}
// CHECK-CC1: {TypedText protocol}{HorizontalSpace  }{Placeholder protocol}

// RUN: c-index-test -code-completion-at=%s:3:2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText end}
// CHECK-CC2: {TypedText optional}
// CHECK-CC2: {TypedText property}
// CHECK-CC2: {TypedText required}

// RUN: c-index-test -code-completion-at=%s:6:2 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: {TypedText dynamic}{HorizontalSpace  }{Placeholder property}
// CHECK-CC3: {TypedText end}
// CHECK-CC3: {TypedText synthesize}{HorizontalSpace  }{Placeholder property}

// RUN: c-index-test -code-completion-at=%s:2:1 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: NotImplemented:{TypedText @class}{HorizontalSpace  }{Placeholder identifier}{SemiColon ;}
// CHECK-CC4: NotImplemented:{TypedText @compatibility_alias}{HorizontalSpace  }{Placeholder alias}{HorizontalSpace  }{Placeholder class}
// CHECK-CC4: NotImplemented:{TypedText @implementation}{HorizontalSpace  }{Placeholder class}
// CHECK-CC4: NotImplemented:{TypedText @interface}{HorizontalSpace  }{Placeholder class}
// CHECK-CC4: NotImplemented:{TypedText @protocol}{HorizontalSpace  }{Placeholder protocol}
// CHECK-CC4: TypedefDecl:{TypedText Class}
// CHECK-CC4: TypedefDecl:{TypedText id}
// CHECK-CC4: TypedefDecl:{TypedText SEL}
// CHECK-CC4: NotImplemented:{TypedText _Bool}

// RUN: c-index-test -code-completion-at=%s:3:1 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: {TypedText @end}
// CHECK-CC5: {TypedText @optional}
// CHECK-CC5: {TypedText @property}
// CHECK-CC5: {TypedText @required}
// CHECK-CC5: TypedefDecl:{TypedText Class}
// CHECK-CC5: TypedefDecl:{TypedText id}
// CHECK-CC5: ObjCInterfaceDecl:{TypedText MyClass}
// CHECK-CC5: TypedefDecl:{TypedText SEL}
// CHECK-CC5: NotImplemented:{TypedText _Bool}

// RUN: c-index-test -code-completion-at=%s:2:23 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: NotImplemented:{TypedText package}
// CHECK-CC6: NotImplemented:{TypedText private}
// CHECK-CC6: NotImplemented:{TypedText protected}
// CHECK-CC6: NotImplemented:{TypedText public}

// RUN: c-index-test -code-completion-at=%s:2:22 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: NotImplemented:{TypedText @package}
// CHECK-CC7: NotImplemented:{TypedText @private}
// CHECK-CC7: NotImplemented:{TypedText @protected}
// CHECK-CC7: NotImplemented:{TypedText @public}
// CHECK-CC7: NotImplemented:{TypedText _Bool}
