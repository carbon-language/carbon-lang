// RUN: %clang_cc1 -emit-llvm-only  -triple i386-apple-darwin9 %s
// rdar://8823265

#pragma pack(1)
struct _two_ms {
        short m:9;      // size is 2
        int q:27;       // size is 6
        short w:13;     // size is 8
        short e:3;      // size is 8
        char r:4;       // size is 9
        char t:7;       // size is 10
        short y:16;     // size is 12
// clang and gcc start differing here. clang seems to follow the rules.
        short u:1;      // size is clang: 13 gcc:14 
        char i:2;       // size is 14
        int a;          // size is 18
        char o:6;       // size is 19
        char s:2;       // size is 19
        short d:10;     // size is 21
        short f:4;      // size is 21
        char b;         // size is 22
        char g:1;       // size is 23
        short h:13;     // size is 25
        char j:8;       // size is 26
        char k:5;       // size is 27
        char c;         // size is 28
        int l:28;       // size is 32
        char z:7;       // size is 33
        int x:20;       // size is clang: 36 gcc:38
        } __attribute__((__ms_struct__));
typedef struct _two_ms two_ms;

// gcc says size is 38, but its does not seem right!
static int a1[(sizeof(two_ms) == 36) - 1];
