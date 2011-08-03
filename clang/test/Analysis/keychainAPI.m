// RUN: %clang_cc1 -analyze -analyzer-checker=experimental.osx.KeychainAPI %s -verify

// Fake typedefs.
typedef unsigned int OSStatus;
typedef unsigned int SecKeychainAttributeList;
typedef unsigned int SecKeychainItemRef;
typedef unsigned int SecItemClass;
typedef unsigned int UInt32;
typedef unsigned int CFTypeRef;
typedef unsigned int UInt16;
typedef unsigned int SecProtocolType;
typedef unsigned int SecAuthenticationType;
enum {
  noErr                      = 0,
  GenericError               = 1
};

// Functions that allocate data.
OSStatus SecKeychainItemCopyContent (
    SecKeychainItemRef itemRef,
    SecItemClass *itemClass,
    SecKeychainAttributeList *attrList,
    UInt32 *length,
    void **outData
);
OSStatus SecKeychainFindGenericPassword (
    CFTypeRef keychainOrArray,
    UInt32 serviceNameLength,
    const char *serviceName,
    UInt32 accountNameLength,
    const char *accountName,
    UInt32 *passwordLength,
    void **passwordData,
    SecKeychainItemRef *itemRef
);
OSStatus SecKeychainFindInternetPassword (
    CFTypeRef keychainOrArray,
    UInt32 serverNameLength,
    const char *serverName,
    UInt32 securityDomainLength,
    const char *securityDomain,
    UInt32 accountNameLength,
    const char *accountName,
    UInt32 pathLength,
    const char *path,
    UInt16 port,
    SecProtocolType protocol,
    SecAuthenticationType authenticationType,
    UInt32 *passwordLength,
    void **passwordData,
    SecKeychainItemRef *itemRef
);

// Function which frees data.
OSStatus SecKeychainItemFreeContent (
    SecKeychainAttributeList *attrList,
    void *data
);

int foo () {
  unsigned int *ptr = 0;
  OSStatus st = 0;

  UInt32 length;
  void *outData;

  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData);
  if (st == noErr)
    SecKeychainItemFreeContent(ptr, outData);

  return 0;
}
