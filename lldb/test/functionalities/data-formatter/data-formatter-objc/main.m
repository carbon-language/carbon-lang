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
    
    MyClass *object = [[MyClass alloc] initWithInt:1 andFloat:3.14 andChar: 'E'];
    
    [object doIncrementByInt:3];
    
    MyOtherClass *object2 = [[MyOtherClass alloc] initWithInt:2 andFloat:6.28 andChar: 'G' andOtherInt:-1];
    
    [object2 doIncrementByInt:3];
    
    NSString *str = [NSString stringWithCString:"A rather short ASCII NSString object is here" encoding:NSASCIIStringEncoding];
    
    NSString *str2 = [NSString stringWithUTF8String:"A rather short UTF8 NSString object is here"];
    
    NSString *str3 = @"A string made with the at sign is here";
    
    NSString *str4 = [NSString stringWithFormat:@"This is string number %ld right here", (long)4];
    
    NSRect rect = {{1,1},{5,5}};
    
    NSString* str5 = NSStringFromRect(rect);
    
    NSString* str6 = [@"/usr/doc/README.1ST" pathExtension];
    
    const unichar myCharacters[] = {0x03C3,'x','x'};
    NSString *str7 = [NSString stringWithCharacters: myCharacters
                                                 length: sizeof myCharacters / sizeof *myCharacters];
        
    NSString* str8 = [@"/usr/doc/file.hasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTime" pathExtension];
    
    const unichar myOtherCharacters[] = {'a',' ', 'v','e','r','y',' ',
    'm','u','c','h',' ','b','o','r','i','n','g',' ','t','a','s','k',
    ' ','t','o',' ','w','r','i','t','e', ' ', 'a', ' ', 's', 't', 'r', 'i', 'n', 'g', ' ',
    't','h','i','s',' ','w','a','y','!','!',0x03C3, 0};
    NSString *str9 = [NSString stringWithCharacters: myOtherCharacters
                                             length: sizeof myOtherCharacters / sizeof *myOtherCharacters];
    
    const unichar myNextCharacters[] = {0x03C3, 0x0000};
    
    NSString *str10 = [NSString stringWithFormat:@"This is a Unicode string %S number %ld right here", myNextCharacters, (long)4];
    
    NSString *str11 = [str10 className];
    
    NSString *label1 = @"Process Name: ";
    NSString *label2 = @"Process Id: ";
    NSString *processName = [[NSProcessInfo processInfo] processName];
    NSString *processID = [NSString stringWithFormat:@"%d", [[NSProcessInfo processInfo] processIdentifier]];
    NSString *str12 = [NSString stringWithFormat:@"%@ %@ %@ %@", label1, processName, label2, processID];
    
    // Set break point at this line.
    [pool drain];
    return 0;
}

