// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

typedef int __darwin_wchar_t;
typedef __darwin_wchar_t wchar_t;
typedef signed short SQLSMALLINT;
typedef SQLSMALLINT SQLRETURN;
typedef enum
  {
    en_sqlstat_total
  }
  sqlerrmsg_t;
SQLRETURN _iodbcdm_sqlerror( )
{
  wchar_t _sqlState[6] = { L"\0" };
}
