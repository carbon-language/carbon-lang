// RUN: clang-cc -triple x86_64-apple-darwin10 -fblocks -fnext-runtime -fobjc-gc -fobjc-newgc-api -emit-llvm -o %t %s &&
// RUN: grep -F '@objc_assign_strongCast' %t  | count 4 &&
// RUN: true

@interface DSATextSearch @end

DSATextSearch **_uniqueIdToIdentifierArray = ((void *)0);
void foo (int _nextId)
{
	_uniqueIdToIdentifierArray[_nextId] = 0;  // objc_assign_strongCast
}

typedef struct {
    unsigned long state;
    id *itemsPtr;
    void (^bp)();
    unsigned long *mutationsPtr;
    unsigned long extra[5];
} NSFastEnumerationState;

void foo1 (NSFastEnumerationState * state)
{
   state->itemsPtr = 0;
   state->bp = ^{};
}

