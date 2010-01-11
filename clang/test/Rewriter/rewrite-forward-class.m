// RUN: %clang_cc1 -rewrite-objc -o - %s
// rdar://6969189

@class XX;
@class YY, ZZ, QQ;
@class ISyncClient, SMSession, ISyncManager, ISyncSession, SMDataclassInfo, SMClientInfo,
    DMCConfiguration, DMCStatusEntry;

