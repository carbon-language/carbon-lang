set(LLVM_NO_RTTI 1)

add_lldb_library(lldbDataFormatters
  CXXFormatterFunctions.cpp
  DataVisualization.cpp
  FormatCache.cpp
  FormatClasses.cpp
  FormatManager.cpp
  TypeCategory.cpp
  TypeCategoryMap.cpp
  TypeFormat.cpp
  TypeSummary.cpp
  TypeSynthetic.cpp
  )
