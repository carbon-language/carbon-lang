// RUN: clang-cc -fvisibility=hidden -triple x86_64-apple-darwin10  -S -o - %s | grep -e "private_extern _OBJC_" | count 2 

@interface INTF @end

@implementation INTF @end

