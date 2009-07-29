// RUN: clang-cc -analyze -checker-cfref -analyzer-store=basic -verify %s &&
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s

typedef const struct __CFAllocator * CFAllocatorRef;
typedef struct __CFDictionary * CFMutableDictionaryRef;
typedef signed long CFIndex;
typedef CFIndex CFNumberType;
typedef const void * CFTypeRef;
typedef struct {} CFDictionaryKeyCallBacks, CFDictionaryValueCallBacks;
typedef const struct __CFNumber * CFNumberRef;
extern const CFAllocatorRef kCFAllocatorDefault;
extern const CFDictionaryKeyCallBacks kCFTypeDictionaryKeyCallBacks;
extern const CFDictionaryValueCallBacks kCFTypeDictionaryValueCallBacks;
enum { kCFNumberSInt32Type = 3 };
CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef allocator, CFIndex capacity, const CFDictionaryKeyCallBacks *keyCallBacks, const CFDictionaryValueCallBacks *valueCallBacks);
void CFDictionaryAddValue(CFMutableDictionaryRef theDict, const void *key, const void *value);
void CFRelease(CFTypeRef cf);
CFTypeRef CFRetain(CFTypeRef cf);
extern CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);
typedef const struct __CFArray * CFArrayRef;
typedef struct __CFArray * CFMutableArrayRef;
void CFArrayAppendValue(CFMutableArrayRef theArray, const void *value);

void f(CFMutableDictionaryRef y, void* key, void* val_key) {
  CFMutableDictionaryRef x = CFDictionaryCreateMutable(kCFAllocatorDefault, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFDictionaryAddValue(y, key, x);
  CFRelease(x); // the dictionary keeps a reference, so the object isn't deallocated yet
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
  if (value) {
    CFDictionaryAddValue(x, val_key, value); // no-warning
    CFRelease(value);
    CFDictionaryAddValue(y, val_key, value); // no-warning
  }
}

// <rdar://problem/6560661>
// Same issue, except with "AppendValue" functions.
void f2(CFMutableArrayRef x) {
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z);
  // CFArrayAppendValue keeps a reference to value.
  CFArrayAppendValue(x, value);
  CFRelease(value);
  CFRetain(value);
  CFRelease(value); // no-warning
}
