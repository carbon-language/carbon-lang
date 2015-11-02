//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface Foo: NSObject
{}
- (id) init;
@end

@interface Bar: Foo
{
	int _iVar;
}
- (id) init;
@end

@implementation Foo

- (id) init
{
	self = [super init];
	return self;
}

@end

@implementation Bar

- (id) init
{
	self = [super init];
	if (self)
		self->_iVar = 5;
	return self;
}

@end

int main()
{
	Bar* aBar = [Bar new];
	id nothing = [aBar noSuchSelector]; // Break at this line
	return 0;
}

