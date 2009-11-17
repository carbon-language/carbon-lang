// RUN: clang-cc -fvisibility=hidden -fobjc-nonfragile-abi -S -o - %s | grep -e "private_extern _OBJC_" | count 2 

@interface INTF @end

@implementation INTF @end

