// RUN: %llvmgcc -c -w -m64 %s -o /dev/null

@class NSDictionary, DSoBuffer, DSoDirectory, NSMutableArray;
@interface NSException {}
@end
@interface DSoNode {
  DSoDirectory  *mDirectory;
}
@end
@implementation DSoNode
- (void) _findRecordsOfTypes {
  DSoBuffer      *dbData;
  void           *recInfo;
  NSMutableArray *results;
  @try {
    dsGetRecordEntry([dbData dsDataBuffer], (void**)&recInfo);
    @try {
        [results addObject:37];
    } @finally {
      dsDeallocRecordEntry([mDirectory dsDirRef], recInfo);
    }
  } @catch(NSException * exception) {
  }
}


