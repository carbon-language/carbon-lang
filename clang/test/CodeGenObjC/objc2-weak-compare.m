// RUN: clang-cc -triple i386-apple-darwin9 -fobjc-gc -emit-llvm -o %t %s

@interface PBXTarget 
{

PBXTarget * __weak _lastKnownTarget;
PBXTarget * __weak _KnownTarget;
PBXTarget * result;
}
- Meth;
@end

@implementation PBXTarget
- Meth {
	if (_lastKnownTarget != result)
	 foo();
	if (result != _lastKnownTarget)
	 foo();

 	if (_lastKnownTarget != _KnownTarget)
	  foo();
}

@end
