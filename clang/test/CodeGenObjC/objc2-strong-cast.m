// RUN: clang-cc -fnext-runtime -fobjc-gc -emit-llvm -o %t %s

@interface I {
  __attribute__((objc_gc(strong))) signed long *_documentIDs;
  __attribute__((objc_gc(strong))) id *IdocumentIDs;
}
- (void) _getResultsOfMatches;
@end

@implementation I
-(void) _getResultsOfMatches {
    _documentIDs[2] = _documentIDs[3];
    IdocumentIDs[2] = IdocumentIDs[3];
}

@end

