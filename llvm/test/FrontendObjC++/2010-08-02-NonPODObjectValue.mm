// RUN: not %llvmgcc %s -S -o - |& FileCheck %s
// This tests for a specific diagnostic in LLVM-GCC.
// Clang compiles this correctly with no diagnostic,
// ergo this test will fail with a Clang-based front-end.
class TFENodeVector  {
public:
 TFENodeVector(const TFENodeVector& inNodeVector);
 TFENodeVector();
};

@interface TWindowHistoryEntry  {}
@property (assign, nonatomic) TFENodeVector targetPath;
@end

@implementation TWindowHistoryEntry
@synthesize targetPath;
- (void) initWithWindowController {
   TWindowHistoryEntry* entry;
   TFENodeVector newPath;
   // CHECK: setting a C++ non-POD object value is not implemented
#ifdef __clang__
#error setting a C++ non-POD object value is not implemented
#endif
   entry.targetPath = newPath;
   [entry setTargetPath:newPath];
}
@end
