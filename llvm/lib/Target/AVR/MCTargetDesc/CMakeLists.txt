add_llvm_library(LLVMAVRDesc
  AVRELFStreamer.cpp
  AVRTargetStreamer.cpp
)

add_dependencies(LLVMAVRDesc AVRCommonTableGen)

