// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -std=c++0x -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s &&
// RUN: true

class UnicodeString {
public:
        enum EInvariant { kInvariant };
        int extract(int targetCapacity, enum EInvariant inv) const;
        int extract(unsigned targetLength, const char *codepage) const;
};

void foo(const UnicodeString& id) {
        enum {BUFLEN = 128 };
        id.extract(BUFLEN - 2, UnicodeString::kInvariant);
}

// CHECK-LP64: call     __ZNK13UnicodeString7extractEiNS_10EInvariantE

// CHECK-LP32: call     L__ZNK13UnicodeString7extractEiNS_10EInvariantE
