// RUN: clang-cc  -fsyntax-only -verify %s

@interface SSyncCEList
{
	id _list;
}
@end

@implementation SSyncCEList

- (id) list { return 0; }
@end

@interface SSyncConflictList : SSyncCEList
@end

@implementation SSyncConflictList

- (id)Meth : (SSyncConflictList*)other
  {
    return other.list;
  }
@end

