// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-extensions

// Wide character predefined identifiers
#define _STR2WSTR(str) L##str
#define STR2WSTR(str) _STR2WSTR(str)
void abcdefghi12(void) {
 const wchar_t (*ss)[12] = &STR2WSTR(__FUNCTION__);
 static int arr[sizeof(STR2WSTR(__FUNCTION__))==12*sizeof(wchar_t) ? 1 : -1];
}
