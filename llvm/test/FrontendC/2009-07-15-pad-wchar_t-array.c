// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

#include <stddef.h>
signed short _iodbcdm_sqlerror( )
{
  wchar_t _sqlState[6] = { L"\0" };
}
