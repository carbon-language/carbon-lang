// RUN: clang-tidy %s --config-file=%S/Inputs/readability-identifier-naming/hungarian-notation1/.clang-tidy 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK-MESSAGES %s

// clang-format off
typedef signed char         int8_t;     // NOLINT
typedef short               int16_t;    // NOLINT
typedef long                int32_t;    // NOLINT
typedef long long           int64_t;    // NOLINT
typedef unsigned char       uint8_t;    // NOLINT
typedef unsigned short      uint16_t;   // NOLINT
typedef unsigned long       uint32_t;   // NOLINT
typedef unsigned long long  uint64_t;   // NOLINT
#ifndef _WIN32
typedef unsigned long long  size_t;     // NOLINT
#endif
typedef long                intptr_t;   // NOLINT
typedef unsigned long       uintptr_t;  // NOLINT
typedef long int            ptrdiff_t;  // NOLINT
typedef unsigned char       BYTE;       // NOLINT
typedef unsigned short      WORD;       // NOLINT
typedef unsigned long       DWORD;      // NOLINT
typedef int                 BOOL;       // NOLINT
typedef int                 BOOLEAN;    // NOLINT
typedef float               FLOAT;      // NOLINT
typedef int                 INT;        // NOLINT
typedef unsigned int        UINT;       // NOLINT
typedef unsigned long       ULONG;      // NOLINT
typedef short               SHORT;      // NOLINT
typedef unsigned short      USHORT;     // NOLINT
typedef char                CHAR;       // NOLINT
typedef unsigned char       UCHAR;      // NOLINT
typedef signed char         INT8;       // NOLINT
typedef signed short        INT16;      // NOLINT
typedef signed int          INT32;      // NOLINT
typedef signed long long    INT64;      // NOLINT
typedef unsigned char       UINT8;      // NOLINT
typedef unsigned short      UINT16;     // NOLINT
typedef unsigned int        UINT32;     // NOLINT
typedef unsigned long long  UINT64;     // NOLINT
typedef long                LONG;       // NOLINT
typedef signed int          LONG32;     // NOLINT
typedef unsigned int        ULONG32;    // NOLINT
typedef uint64_t            ULONG64;    // NOLINT
typedef unsigned int        DWORD32;    // NOLINT
typedef uint64_t            DWORD64;    // NOLINT
typedef uint64_t            ULONGLONG;  // NOLINT
typedef void*               PVOID;      // NOLINT
typedef void*               HANDLE;     // NOLINT
typedef void*               FILE;       // NOLINT
#define NULL                (0)         // NOLINT
// clang-format on

// clang-format off
//===----------------------------------------------------------------------===//
// Cases to CheckOptions
//===----------------------------------------------------------------------===//
class CMyClass1 {
public:
  static int ClassMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for class member 'ClassMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  static int iClassMemberCase;

  char const ConstantMemberCase = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for constant member 'ConstantMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char const cConstantMemberCase = 0;

  void MyFunc1(const int ConstantParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid case style for constant parameter 'ConstantParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void MyFunc1(const int iConstantParameterCase);

  void MyFunc2(const int* ConstantPointerParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: invalid case style for pointer parameter 'ConstantPointerParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void MyFunc2(const int* piConstantPointerParameterCase);

  static constexpr int ConstexprVariableCase = 123;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for constexpr variable 'ConstexprVariableCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  static constexpr int iConstexprVariableCase = 123;
};

const int GlobalConstantCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global constant 'GlobalConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int iGlobalConstantCase = 0;

const int* GlobalConstantPointerCase = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global pointer 'GlobalConstantPointerCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int* piGlobalConstantPointerCase = nullptr;

int* GlobalPointerCase = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global pointer 'GlobalPointerCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int* piGlobalPointerCase = nullptr;

int GlobalVariableCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'GlobalVariableCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iGlobalVariableCase = 0;

void Func1(){
  int const LocalConstantCase = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for local constant 'LocalConstantCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int const iLocalConstantCase = 3;

  unsigned const ConstantCase = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for local constant 'ConstantCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  unsigned const uConstantCase = 1;

  int* const LocalConstantPointerCase = nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for local constant pointer 'LocalConstantPointerCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int* const piLocalConstantPointerCase = nullptr;

  int *LocalPointerCase = nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for local pointer 'LocalPointerCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int *piLocalPointerCase = nullptr;

  int LocalVariableCase = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable 'LocalVariableCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int iLocalVariableCase = 0;
}

class CMyClass2 {
  char MemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for private member 'MemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char cMemberCase;

  void Func1(int ParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for parameter 'ParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func1(int iParameterCase);

  void Func2(const int ParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for constant parameter 'ParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func2(const int iParameterCase);

  void Func3(const int *PointerParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid case style for pointer parameter 'PointerParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func3(const int *piPointerParameterCase);
};

class CMyClass3 {
private:
  char PrivateMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for private member 'PrivateMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char cPrivateMemberCase;

protected:
  char ProtectedMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for protected member 'ProtectedMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char cProtectedMemberCase;

public:
  char PublicMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for public member 'PublicMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char cPublicMemberCase;
};

static const int StaticConstantCase = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global constant 'StaticConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static const int iStaticConstantCase = 3;

static int StaticVariableCase = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'StaticVariableCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static int iStaticVariableCase = 3;

struct MyStruct { int StructCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: invalid case style for public member 'StructCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}struct MyStruct { int iStructCase; };

union MyUnion { int UnionCase; long lUnionCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for public member 'UnionCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}union MyUnion { int iUnionCase; long lUnionCase; };

//===----------------------------------------------------------------------===//
// C string
//===----------------------------------------------------------------------===//
const char *NamePtr = "Name";
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global pointer 'NamePtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char *szNamePtr = "Name";

const char NameArray[] = "Name";
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global constant 'NameArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char szNameArray[] = "Name";

const char *NamePtrArray[] = {"AA", "BB"};
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'NamePtrArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char *pszNamePtrArray[] = {"AA", "BB"};

const wchar_t *WideNamePtr = L"Name";
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'WideNamePtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t *wszWideNamePtr = L"Name";

const wchar_t WideNameArray[] = L"Name";
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global constant 'WideNameArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t wszWideNameArray[] = L"Name";

const wchar_t *WideNamePtrArray[] = {L"AA", L"BB"};
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global variable 'WideNamePtrArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t *pwszWideNamePtrArray[] = {L"AA", L"BB"};

class CMyClass4 {
private:
  char *Name = "Text";
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for private member 'Name' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char *szName = "Text";

  const char *ConstName = "Text";
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for private member 'ConstName' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  const char *szConstName = "Text";

public:
  const char* DuplicateString(const char* Input, size_t nRequiredSize);
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: invalid case style for pointer parameter 'Input' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  const char* DuplicateString(const char* szInput, size_t nRequiredSize);

  size_t UpdateText(const char* Buffer, size_t nBufferSize);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: invalid case style for pointer parameter 'Buffer' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  size_t UpdateText(const char* szBuffer, size_t nBufferSize);
};


//===----------------------------------------------------------------------===//
// Microsoft Windows data types
//===----------------------------------------------------------------------===//
DWORD MsDword = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsDword' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD dwMsDword = 0;

BYTE MsByte = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsByte' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BYTE byMsByte = 0;

WORD MsWord = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsWord' [readability-identifier-naming]
// CHECK-FIXES: {{^}}WORD wMsWord = 0;

BOOL MsBool = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsBool' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BOOL bMsBool = 0;

BOOLEAN MsBoolean = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsBoolean' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BOOLEAN bMsBoolean = 0;

CHAR MsValueChar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueChar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}CHAR cMsValueChar = 0;

UCHAR MsValueUchar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUchar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UCHAR ucMsValueUchar = 0;

SHORT MsValueShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}SHORT sMsValueShort = 0;

USHORT MsValueUshort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUshort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}USHORT usMsValueUshort = 0;

WORD MsValueWord = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueWord' [readability-identifier-naming]
// CHECK-FIXES: {{^}}WORD wMsValueWord = 0;

DWORD MsValueDword = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueDword' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD dwMsValueDword = 0;

DWORD32 MsValueDword32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueDword32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD32 dw32MsValueDword32 = 0;

DWORD64 MsValueDword64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueDword64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD64 dw64MsValueDword64 = 0;

LONG MsValueLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}LONG lMsValueLong = 0;

ULONG MsValueUlong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUlong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG ulMsValueUlong = 0;

ULONG32 MsValueUlong32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueUlong32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG32 ul32MsValueUlong32 = 0;

ULONG64 MsValueUlong64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueUlong64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG64 ul64MsValueUlong64 = 0;

ULONGLONG MsValueUlongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'MsValueUlongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONGLONG ullMsValueUlongLong = 0;

HANDLE MsValueHandle = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'MsValueHandle' [readability-identifier-naming]
// CHECK-FIXES: {{^}}HANDLE hMsValueHandle = 0;

INT MsValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'MsValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT iMsValueInt = 0;

INT8 MsValueInt8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueInt8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT8 i8MsValueInt8 = 0;

INT16 MsValueInt16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueInt16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT16 i16MsValueInt16 = 0;

INT32 MsValueInt32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueInt32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT32 i32MsValueInt32 = 0;

INT64 MsValueINt64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueINt64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT64 i64MsValueINt64 = 0;

UINT MsValueUint = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueUint' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT uiMsValueUint = 0;

UINT8 MsValueUint8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUint8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT8 u8MsValueUint8 = 0;

UINT16 MsValueUint16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT16 u16MsValueUint16 = 0;

UINT32 MsValueUint32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT32 u32MsValueUint32 = 0;

UINT64 MsValueUint64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT64 u64MsValueUint64 = 0;

PVOID MsValuePvoid = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'MsValuePvoid' [readability-identifier-naming]
// CHECK-FIXES: {{^}}PVOID pMsValuePvoid = NULL;


//===----------------------------------------------------------------------===//
// Array
//===----------------------------------------------------------------------===//
unsigned GlobalUnsignedArray[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'GlobalUnsignedArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned aryGlobalUnsignedArray[] = {1, 2, 3};

int GlobalIntArray[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'GlobalIntArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int aryGlobalIntArray[] = {1, 2, 3};

int DataInt[1] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'DataInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int aryDataInt[1] = {0};

int DataArray[2] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'DataArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int aryDataArray[2] = {0};


//===----------------------------------------------------------------------===//
// Pointer
//===----------------------------------------------------------------------===//
int *DataIntPtr[1] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'DataIntPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int *paryDataIntPtr[1] = {0};

void *BufferPtr1;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'BufferPtr1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void *pBufferPtr1;

void **BufferPtr2;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'BufferPtr2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void **ppBufferPtr2;

void **pBufferPtr3;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'pBufferPtr3' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void **ppBufferPtr3;

int *pBufferPtr4;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global pointer 'pBufferPtr4' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int *piBufferPtr4;

typedef void (*FUNC_PTR_HELLO)();
FUNC_PTR_HELLO Hello = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'Hello' [readability-identifier-naming]
// CHECK-FIXES: {{^}}FUNC_PTR_HELLO fnHello = NULL;

void *ValueVoidPtr = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'ValueVoidPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void *pValueVoidPtr = NULL;

ptrdiff_t PtrDiff = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'PtrDiff' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ptrdiff_t pPtrDiff = NULL;

int8_t *ValueI8Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global pointer 'ValueI8Ptr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int8_t *pi8ValueI8Ptr;

uint8_t *ValueU8Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global pointer 'ValueU8Ptr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint8_t *pu8ValueU8Ptr;

void MyFunc2(void* Val){}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for pointer parameter 'Val' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void MyFunc2(void* pVal){}


//===----------------------------------------------------------------------===//
// Reference
//===----------------------------------------------------------------------===//
int iValueIndex = 1;
int &RefValueIndex = iValueIndex;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'RefValueIndex' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int &iRefValueIndex = iValueIndex;

const int &ConstRefValue = iValueIndex;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ConstRefValue' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int &iConstRefValue = iValueIndex;

long long llValueLongLong = 2;
long long &RefValueLongLong = llValueLongLong;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'RefValueLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long &llRefValueLongLong = llValueLongLong;


//===----------------------------------------------------------------------===//
// Various types
//===----------------------------------------------------------------------===//
int8_t ValueI8;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueI8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int8_t i8ValueI8;

int16_t ValueI16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int16_t i16ValueI16 = 0;

int32_t ValueI32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int32_t i32ValueI32 = 0;

int64_t ValueI64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int64_t i64ValueI64 = 0;

uint8_t ValueU8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueU8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint8_t u8ValueU8 = 0;

uint16_t ValueU16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint16_t u16ValueU16 = 0;

uint32_t ValueU32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint32_t u32ValueU32 = 0;

uint64_t ValueU64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint64_t u64ValueU64 = 0;

float ValueFloat = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ValueFloat' [readability-identifier-naming]
// CHECK-FIXES: {{^}}float fValueFloat = 0;

double ValueDouble = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueDouble' [readability-identifier-naming]
// CHECK-FIXES: {{^}}double dValueDouble = 0;

char ValueChar = 'c';
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueChar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}char cValueChar = 'c';

bool ValueBool = true;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueBool' [readability-identifier-naming]
// CHECK-FIXES: {{^}}bool bValueBool = true;

int ValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'ValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iValueInt = 0;

size_t ValueSize = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueSize' [readability-identifier-naming]
// CHECK-FIXES: {{^}}size_t nValueSize = 0;

wchar_t ValueWchar = 'w';
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueWchar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}wchar_t wcValueWchar = 'w';

short ValueShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ValueShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}short sValueShort = 0;

unsigned ValueUnsigned = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueUnsigned' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned uValueUnsigned = 0;

signed ValueSigned = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueSigned' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed sValueSigned = 0;

long ValueLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long lValueLong = 0;

long long ValueLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'ValueLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long llValueLongLong = 0;

long long int ValueLongLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueLongLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long int lliValueLongLongInt = 0;

long double ValueLongDouble = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'ValueLongDouble' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long double ldValueLongDouble = 0;

signed int ValueSignedInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ValueSignedInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed int siValueSignedInt = 0;

signed short ValueSignedShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'ValueSignedShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed short ssValueSignedShort = 0;

signed short int ValueSignedShortInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ValueSignedShortInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed short int ssiValueSignedShortInt = 0;

signed long long ValueSignedLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ValueSignedLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long long sllValueSignedLongLong = 0;

signed long int ValueSignedLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for global variable 'ValueSignedLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long int sliValueSignedLongInt = 0;

signed long ValueSignedLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'ValueSignedLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long slValueSignedLong = 0;

unsigned long long int ValueUnsignedLongLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for global variable 'ValueUnsignedLongLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long long int ulliValueUnsignedLongLongInt = 0;

unsigned long long ValueUnsignedLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global variable 'ValueUnsignedLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long long ullValueUnsignedLongLong = 0;

unsigned long int ValueUnsignedLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for global variable 'ValueUnsignedLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long int uliValueUnsignedLongInt = 0;

unsigned long ValueUnsignedLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueUnsignedLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long ulValueUnsignedLong = 0;

unsigned short int ValueUnsignedShortInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global variable 'ValueUnsignedShortInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned short int usiValueUnsignedShortInt = 0;

unsigned short ValueUnsignedShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global variable 'ValueUnsignedShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned short usValueUnsignedShort = 0;

unsigned int ValueUnsignedInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'ValueUnsignedInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned int uiValueUnsignedInt = 0;

long int ValueLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long int liValueLongInt = 0;


//===----------------------------------------------------------------------===//
// Specifier, Qualifier, Other keywords
//===----------------------------------------------------------------------===//
volatile int VolatileInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'VolatileInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}volatile int iVolatileInt = 0;

thread_local int ThreadLocalValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ThreadLocalValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}thread_local int iThreadLocalValueInt = 0;

extern int ExternValueInt;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ExternValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}extern int iExternValueInt;

struct DataBuffer {
    mutable size_t Size;
};
// CHECK-MESSAGES: :[[@LINE-2]]:20: warning: invalid case style for public member 'Size' [readability-identifier-naming]
// CHECK-FIXES: {{^}}    mutable size_t nSize;

static constexpr int const &ConstExprInt = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: invalid case style for constexpr variable 'ConstExprInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static constexpr int const &iConstExprInt = 42;


//===----------------------------------------------------------------------===//
// Redefined types
//===----------------------------------------------------------------------===//
typedef int INDEX;
INDEX iIndex = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'iIndex' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INDEX Index = 0;


//===----------------------------------------------------------------------===//
// Class
//===----------------------------------------------------------------------===//
class ClassCase { int Func(); };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'ClassCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class CClassCase { int Func(); };

class AbstractClassCase { virtual int Func() = 0; };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for abstract class 'AbstractClassCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class IAbstractClassCase { virtual int Func() = 0; };

class AbstractClassCase1 { virtual int Func1() = 0; int Func2(); };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for abstract class 'AbstractClassCase1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class IAbstractClassCase1 { virtual int Func1() = 0; int Func2(); };

class ClassConstantCase { public: static const int iConstantCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'ClassConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class CClassConstantCase { public: static const int iConstantCase; };

//===----------------------------------------------------------------------===//
// Other Cases
//===----------------------------------------------------------------------===//
int lower_case = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iLowerCase = 0;

int lower_case1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iLowerCase1 = 0;

int lower_case_2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case_2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iLowerCase2 = 0;

int UPPER_CASE = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'UPPER_CASE' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iUpperCase = 0;

int UPPER_CASE_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'UPPER_CASE_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iUpperCase1 = 0;

int camelBack = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelBack = 0;

int camelBack_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelBack1 = 0;

int camelBack2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelBack2 = 0;

int CamelCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelCase = 0;

int CamelCase_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelCase1 = 0;

int CamelCase2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelCase2 = 0;

int camel_Snake_Back = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camel_Snake_Back' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelSnakeBack = 0;

int camel_Snake_Back_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camel_Snake_Back_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelSnakeBack1 = 0;

int Camel_Snake_Case = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Camel_Snake_Case' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelSnakeCase = 0;

int Camel_Snake_Case_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Camel_Snake_Case_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int iCamelSnakeCase1 = 0;

//===----------------------------------------------------------------------===//
// Enum
//===----------------------------------------------------------------------===//
enum REV_TYPE { RevValid };
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for enum constant 'RevValid' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum REV_TYPE { rtRevValid };

enum EnumConstantCase { OneByte, TwoByte };
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid case style for enum constant 'OneByte' [readability-identifier-naming]
// CHECK-MESSAGES: :[[@LINE-2]]:34: warning: invalid case style for enum constant 'TwoByte' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum EnumConstantCase { eccOneByte, eccTwoByte };

enum class ScopedEnumConstantCase { Case1 };
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: invalid case style for scoped enum constant 'Case1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum class ScopedEnumConstantCase { seccCase1 };
// clang-format on
