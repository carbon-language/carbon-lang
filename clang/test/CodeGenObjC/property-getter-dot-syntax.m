// RUN: clang-cc --emit-llvm -o %t %s

@protocol NSObject
- (void *)description;
@end

int main()
{
        id<NSObject> eggs;
        void *eggsText= eggs.description;
}
