// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-nonfragile-abi -fexceptions -o - %s | FileCheck %s
// rdar://8409336

struct TFENode {
void GetURL() const;
};

@interface TNodeIconAndNameCell
- (const TFENode&) node;
@end

@implementation TNodeIconAndNameCell     
- (const TFENode&) node {
// CHECK: call %struct.TFENode* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK-NEXT: call void @_ZNK7TFENode6GetURLEv(%struct.TFENode* %{{.*}})
	self.node.GetURL();
}	// expected-warning {{control reaches end of non-void function}}
@end
