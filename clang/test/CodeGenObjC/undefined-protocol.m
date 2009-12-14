// RUN: clang -cc1 -emit-llvm-only -fgnu-runtime %s

@protocol MadeUpProtocol;

@interface Object <MadeUpProtocol> @end
@implementation Object @end
