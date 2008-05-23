// RUN: clang -rewrite-objc -o - %s
// rdar://5950938
@interface NSArray {}
+ (id)arrayWithObjects:(id)firstObj, ...;
@end

@interface NSConstantString {}
@end

int main() {
    id foo = [NSArray arrayWithObjects:@"1", @"2", @"3", @"4", @"5", @"6", @"7", @"8", @"9", @"10", @"11", @"12", 0];
    return 0;
}

