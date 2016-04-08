set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(merge-fdata
  merge-fdata.cpp
  ../DataReader.cpp
)
