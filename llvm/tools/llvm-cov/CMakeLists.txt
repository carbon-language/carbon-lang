set(LLVM_LINK_COMPONENTS core support object profiledata)

add_llvm_tool(llvm-cov
  llvm-cov.cpp
  gcov.cpp
  CodeCoverage.cpp
  CoverageFilters.cpp
  CoverageReport.cpp
  CoverageSummary.cpp
  CoverageSummaryInfo.cpp
  SourceCoverageView.cpp
  TestingSupport.cpp
  )
