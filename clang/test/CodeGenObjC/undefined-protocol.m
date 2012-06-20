// RUN: %clang_cc1 -emit-llvm-only -fobjc-runtime=gnu %s

@protocol MadeUpProtocol;

@interface Object <MadeUpProtocol> @end
@implementation Object @end
