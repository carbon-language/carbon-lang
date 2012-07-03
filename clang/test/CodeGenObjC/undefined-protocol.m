// RUN: %clang_cc1 -emit-llvm-only -fobjc-runtime=gcc %s

@protocol MadeUpProtocol;

@interface Object <MadeUpProtocol> @end
@implementation Object @end
