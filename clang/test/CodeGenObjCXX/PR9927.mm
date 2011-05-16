// RUN: %clang_cc1 -emit-llvm-only %s

// Test that we don't crash.

class allocator {
};
class basic_string     {
struct _Alloc_hider : allocator       {
char* _M_p;
};
_Alloc_hider _M_dataplus;
};
@implementation
CrashReporterUI -(void)awakeFromNib {
}
-(void)showCrashUI:(const basic_string&)dumpfile   {
}
@end
