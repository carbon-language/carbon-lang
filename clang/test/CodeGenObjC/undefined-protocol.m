// RUN: clang-cc -emit-llvm-only -fgnu-runtime %s

@protocol MadeUpProtocol;

@interface Object <MadeUpProtocol> @end
@implementation Object @end
