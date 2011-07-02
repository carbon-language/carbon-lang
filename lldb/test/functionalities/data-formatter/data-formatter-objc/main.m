//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface MyClass : NSObject
{
    int i;
    char c;
    float f; 
}

- (id)initWithInt: (int)x andFloat:(float)y andChar:(char)z;
- (int)doIncrementByInt: (int)x;

@end

@interface MyOtherClass : MyClass
{
    int i2;
    MyClass *backup;
}
- (id)initWithInt: (int)x andFloat:(float)y andChar:(char)z andOtherInt:(int)q;

@end

@implementation MyClass

- (id)initWithInt: (int)x andFloat:(float)y andChar:(char)z
{
    self = [super init];
    if (self) {
        self->i = x;
        self->f = y;
        self->c = z;
    }    
    return self;
}

- (int)doIncrementByInt: (int)x
{
    self->i += x;
    return self->i;
}

@end

@implementation MyOtherClass

- (id)initWithInt: (int)x andFloat:(float)y andChar:(char)z andOtherInt:(int)q
{
    self = [super initWithInt:x andFloat:y andChar:z];
    if (self) {
        self->i2 = q;
        self->backup = [[MyClass alloc] initWithInt:x andFloat:y andChar:z];
    }    
    return self;
}

@end

int main (int argc, const char * argv[])
{
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    
    // insert code here...
    NSLog(@"Hello, World!");
    
    MyClass *object = [[MyClass alloc] initWithInt:1 andFloat:3.14 andChar: 'E'];
    
    [object doIncrementByInt:3];
    
    MyOtherClass *object2 = [[MyOtherClass alloc] initWithInt:2 andFloat:6.28 andChar: 'G' andOtherInt:-1];
    
    [object2 doIncrementByInt:3];
    // Set break point at this line.
    [pool drain];
    return 0;
}

