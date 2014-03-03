#pragma once

#if defined (_MSC_VER) 
#   if defined(EXPORT_LIBLLDB)
#       define  LLDB_API __declspec(dllexport)
#   elif defined(IMPORT_LIBLLDB)
#       define  LLDB_API __declspec(dllimport)
#   else
#       define LLDB_API
#   endif
#else /* defined (_MSC_VER) */
#   define LLDB_API
#endif

