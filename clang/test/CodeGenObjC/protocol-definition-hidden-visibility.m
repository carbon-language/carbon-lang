// RUN: clang-cc -triple x86_64-apple-darwin10  -S -o - %s | grep -e "private_extern l_OBJC_PROTOCOL_" | count 2

@interface FOO @end

@interface NSObject @end

@protocol SSHIPCProtocolHandler_BDC;

typedef NSObject<SSHIPCProtocolHandler_BDC> _SSHIPCProtocolHandler_BDC;

@interface SSHIPC_v2_RPFSProxy
@property(nonatomic,readonly,retain) _SSHIPCProtocolHandler_BDC* protocolHandler_BDC;
@end

@implementation FOO
- (_SSHIPCProtocolHandler_BDC*) protocolHandler_BDC {@protocol(SSHIPCProtocolHandler_BDC); }
@end


